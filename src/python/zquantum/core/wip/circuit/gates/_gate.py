"""Base classes for implementing quantum gates."""
from abc import abstractmethod, ABC
import inspect
import numpy as np
import sympy
import json
import copy
import warnings
from typing import Tuple, Union, Dict, TextIO, Set, Any, Optional, Iterable
from ....utils import SCHEMA_VERSION


def _evaluate_parameter(param, symbol_map: Dict[sympy.Expr, Any]):
    if isinstance(param, sympy.Symbol):
        return symbol_map.get(param, param)
    elif isinstance(param, sympy.Expr):
        return param.subs(symbol_map)
    else:
        return param


class Gate(ABC):
    """Base class for quantum Gates.

    Args:
        qubits:  tuple of qubit indices this gates acts on.

    Attributes:
        qubits:  tuple of qubit indices this gates acts on.
    """

    def __init__(self, qubits: Tuple[int, ...]):
        if not self.are_qubits_unique(qubits):
            raise ValueError("Qubits need to be unique.")
        self.qubits = qubits

    @property
    @abstractmethod
    def matrix(self) -> sympy.Matrix:
        """Matrix representation of this gate."""
        ...

    @property
    def params(self) -> Tuple[Any, ...]:
        return ()

    @property
    def symbolic_params(self) -> Iterable[str]:
        """Free symbols present in gate's matrix."""
        symbols = set(
            symbol
            for element in self.matrix
            if isinstance(element, sympy.Expr)
            for symbol in element.free_symbols
        )
        return sorted(symbols, key=str)

    @staticmethod
    def is_valid_operator(matrix: sympy.Matrix, qubits: Tuple[int, ...]) -> bool:
        """Check if given matrix represents a gate that can act on specified qubits.

        A matrix is considered a valid operator if its shape is (2^N, ^N) where N=len(qubits)
        and it is unitary. The unitary assumption is checked if and only the matrix does not
        have parameters.

        Args:
            matrix: matrix representing the gate.
            qubits: qubits that gate should act on.
        Returns:
            True if matrix is a valid operator acting on specified qubits and False otherwise.
        """
        return matrix.shape == 2 * (2 ** len(qubits),)

    @staticmethod
    def are_elements_equal(element, another_element) -> bool:
        """Determine if two elements from gates' matrices are equal.

        This is to be used in __eq__ method when comparing matrices elementwise.

        Args:
            element: first value to compare. It can be float, complex or some sympy expression.
            another_element: second value to compare.
        """
        difference = sympy.N(sympy.expand(element) - sympy.expand(another_element))

        try:
            return np.allclose(
                float(sympy.re(difference)) + 1j * float(sympy.im(difference)), 0
            )
        except TypeError:
            return False

    @staticmethod
    def are_qubits_unique(qubits: Tuple[int, ...]):
        """Check if given tuple of qubit numbers contains unique entries.

        Args:
            qubits: tuple of qubit indices to be checked.
        Returns:
            True if all qubits are unique and False otherwise.
        """
        return len(qubits) == len(set(qubits))

    @property
    def is_parameterized(self):
        """Boolean indicating if any symbolic parameters used in the gate's matrix."""
        for element in self.matrix:
            if isinstance(element, sympy.Expr):
                return True
        return False

    def __eq__(self, another_gate):
        """Determine if two gates are equivalent.

        Args:
            another_gate (Gate): The gate with which to compare

        Returns:
            True if self is equivalent with another_gate and False otherwise
        """
        if len(self.qubits) != len(another_gate.qubits):
            return False
        elif any(
            qubit != another_qubit
            for qubit, another_qubit in zip(self.qubits, another_gate.qubits)
        ):
            return False
        elif len(self.matrix) != len(another_gate.matrix):
            return False
        elif self.symbolic_params != another_gate.symbolic_params:
            return False
        elif any(
            not self.are_elements_equal(element, another_element)
            for element, another_element in zip(self.matrix, another_gate.matrix)
        ):
            return False
        return True

    def to_dict(self, serializable: bool = True):
        """Convert the Gate object into a dictionary.

        Args:
            serializable (bool): If true, the returned dictionary is serializable so that it can be stored
                in JSON format

        Returns:
            Dict: keys are schema, qubits, matrix, and symbolic_params.
        """

        gate_dict = {"schema": SCHEMA_VERSION + "-gate"}
        if serializable:
            gate_dict["qubits"] = list(self.qubits)
            gate_dict["matrix"] = [
                {"elements": [str(element) for element in self.matrix.row(i)]}
                for i in range(self.matrix.shape[0])
            ]
            gate_dict["symbolic_params"] = [
                str(param) for param in self.symbolic_params
            ]
        else:
            gate_dict["qubits"] = self.qubits
            gate_dict["matrix"] = self.matrix
            gate_dict["symbolic_params"] = self.symbolic_params
        return gate_dict

    def save(self, filename: str):
        """Save the Gate object to file in JSON format

        Args:
            filename (str): The path to the file to store the Gate
        """
        with open(filename, "w") as f:
            json.dump(self.to_dict(serializable=True), f, indent=2)

    @staticmethod
    def load(data: Union[Dict, TextIO, str]):
        """Load a Gate object from either a file/file-like object or a dictionary

        Args:
            data (Union[Dict, TextIO]): The data to load into the gate object

        Returns:
            CustomGate read from given data source.
        """
        if isinstance(data, str):
            with open(data, "r") as f:
                data = json.load(f)

        elif not isinstance(data, dict):
            data = json.load(data)

        qubits = tuple(data["qubits"])
        symbols = {
            symbol: sympy.Symbol(symbol) if isinstance(symbol, str) else symbol
            for symbol in data["symbolic_params"]
        }
        if not isinstance(data["matrix"], sympy.Matrix):
            matrix = sympy.Matrix(
                [
                    [
                        sympy.sympify(element, locals=symbols)
                        for element in row["elements"]
                    ]
                    for row in data["matrix"]
                ]
            )
        else:
            matrix = data["matrix"]

        return CustomGate(matrix, qubits)

    @property
    def dagger(self) -> "Gate":
        """Return inverse of this gate.

        Note that different subclasses of Gate can implement dagger differently.
        For instance, for Pauli X, Y, Z operators this is an identity operation.
        """
        return Dagger(self)

    def evaluate(self, symbols_map: Dict[sympy.Symbol, Any]) -> "Gate":
        new_params = [_evaluate_parameter(param, symbols_map) for param in self.params]
        return type(self)(*self.qubits, *new_params)

    def __str__(self):
        args_reprs = [str(qubit) for qubit in self.qubits]
        if self.params:
            param_names = [
                param.name
                for param in inspect.signature(self.__init__).parameters.values()
            ][-len(self.params) :]

            args_reprs += [
                f"{param_name}={param_value}".replace(" ", "")
                for param_name, param_value in zip(param_names, self.params)
            ]

        return f"{type(self).__name__}({', '.join(args_reprs)})"

    def __repr__(self):
        return str(self)


class CustomGate(Gate):
    """Gate class with custom matrix."""

    def __init__(
        self, matrix: sympy.Matrix, qubits: Tuple[int, ...], name: Optional[str] = None
    ):
        """Initialize a gate

        Args:
            matrix (sympy.Matrix): See class definition
            qubits (tuple(int)): See class definition
        """
        if not self.is_valid_operator(matrix, qubits):
            raise ValueError(
                f"Passed matrix has shape {matrix.shape} and cannot act on "
                f"{len(qubits)} qubits."
            )

        super().__init__(qubits)
        copied_matrix = copy.deepcopy(matrix)

        self.name = name if name is not None else str(matrix)

        for index, element in enumerate(copied_matrix):
            copied_matrix[index] = element.evalf()

        self._matrix = copied_matrix

    @property
    def matrix(self) -> sympy.Matrix:
        """Matrix representation of this gate."""
        return self._matrix

    def evaluate(self, symbols_map: Dict[str, Any]) -> "Gate":
        """Return a copy of self with symbolic parameters substituted according to provided map.

        Args:
            symbols_map (Dict): A map of the symbols/gate parameters to new values

        Returns:
            A copy of self with the same matrix, but every symbolic param in the map is
            substituted with corresponding value.
        """
        if not self.is_parameterized:
            warnings.warn(
                """Gate is not parameterized. evaluate will return a copy of the current gate."""
            )
            return copy.deepcopy(self)

        if any(symbol not in self.symbolic_params for symbol in symbols_map):
            warnings.warn(
                f"""
                Trying to evaluate gate with symbols not existing in the gate:
                Symbols in circuit: {self.symbolic_params}
                Symbols in the map: {symbols_map.keys()}
                """,
                Warning,
            )

        evaluated_matrix = copy.deepcopy(self.matrix)

        for index, element in enumerate(evaluated_matrix):
            new_element = element.subs(symbols_map).evalf()
            if isinstance(sympy.re(new_element), sympy.Number) and isinstance(
                sympy.im(new_element), sympy.Number
            ):
                new_element = complex(
                    round(sympy.re(new_element), 16), round(sympy.im(new_element), 16)
                )
            evaluated_matrix[index] = new_element

        evaluated_gate = CustomGate(evaluated_matrix, self.qubits)
        return evaluated_gate

    def __repr__(self) -> str:
        return f"zquantum.core.circuit.gate.CustomGate(matrix={self.matrix}, qubits={self.qubits})"


class SpecializedGate(Gate):
    """Base class for known specialized gates (e.g. X, Hadamard, RX etc.)."""

    def __init__(self, qubits):
        self._matrix = None
        super().__init__(qubits)

    @abstractmethod
    def _create_matrix(self) -> sympy.Matrix:
        """Create matrix of this gate."""
        pass

    @property
    def matrix(self) -> sympy.Matrix:
        if self._matrix is None:
            self._matrix = self._create_matrix()
        return self._matrix


class ControlledGate(SpecializedGate):
    """Controlled quantum gate."""

    def __init__(self, target_gate: Gate, control: int):
        super().__init__((control,) + target_gate.qubits)
        self.control = control
        self.target_gate = target_gate

    def evaluate(self, symbols_map: Dict[str, Any]) -> "Gate":
        return ControlledGate(self.target_gate.evaluate(symbols_map), self.control)

    @property
    def dagger(self) -> "Gate":
        return ControlledGate(self.target_gate.dagger, self.control)

    @property
    def params(self):
        return self.target_gate.params

    def _create_matrix(self) -> sympy.Matrix:
        target_matrix = self.target_gate.matrix
        return sympy.Matrix.diag(sympy.eye(target_matrix.shape[0]), target_matrix)


class Dagger(SpecializedGate):
    def __init__(self, gate: Gate):
        super().__init__(gate.qubits)
        self.gate = gate

    def params(self) -> Tuple[Any, ...]:
        return self.gate.params

    def evaluate(self, symbols_map: Dict[str, Any]) -> "Gate":
        return Dagger(self.gate.evaluate(symbols_map))

    @property
    def dagger(self) -> "Gate":
        return self.gate

    def _create_matrix(self) -> sympy.Matrix:
        return self.gate.matrix.conjugate().T


class HermitianMixin:
    """Add this to inheritance hierarchy of your class to make dagger behave as identity.

    For an example application see Pauli gates (X, Y, Z).
    """

    @property
    def dagger(self):
        return self
