"""Class hierarchy for base gates."""
import math
from dataclasses import dataclass
from functools import singledispatch, reduce
from numbers import Number
from typing import Tuple, Union, Callable, Dict, Optional, Iterable, Any

import sympy
from typing_extensions import Protocol

Parameter = Union[sympy.Symbol, Number]


class Gate(Protocol):
    """Quantum gate representable by a matrix, translatable to other frameworks and backends."""

    @property
    def name(self) -> str:
        """Name of the gate.

        Name is used in textual representation and dispatching in conversion between
        frameworks, therefore implementers of this protocol should make sure
        the default gate names are not used.
        """
        raise NotImplementedError()

    @property
    def params(self) -> Tuple[Parameter, ...]:
        """Value of parameters bound to this gate.

        Length of `params` should be equal to number of parameters in gate's initializer.
        In particular, nonparametric gates should always return ().

        We need it for translations to other frameworks.
        """
        raise NotImplementedError()

    @property
    def num_qubits(self) -> int:
        """Number of qubits this gate acts on.
        We need it because matrix is computed lazily, and we don't want to create matrix
        just to know the number of qubits.
        """
        raise NotImplementedError()

    @property
    def matrix(self) -> sympy.Matrix:
        """Unitary matrix describing gate's action on state vector.

        We need it to be able to implement .propagate() on the operation class.
        """
        raise NotImplementedError()

    def controlled(self, num_control_qubits: int) -> "Gate":
        raise NotImplementedError()

    @property
    def dagger(self) -> "Gate":
        raise NotImplementedError()

    def bind(self, symbols_map: Dict[sympy.Symbol, Parameter]) -> "Gate":
        raise NotImplementedError()

    def __call__(self, *qubit_indices: int) -> "GateOperation":
        """Apply this gate on qubits in a circuit."""
        return GateOperation(self, qubit_indices)


@dataclass(frozen=True)
class GateOperation:
    gate: Gate
    qubit_indices: Tuple[int, ...]


@singledispatch
def _sub_symbols(parameter, symbols_map: Dict[sympy.Symbol, Parameter]) -> Parameter:
    raise NotImplementedError()


@_sub_symbols.register
def _sub_symbols_in_number(parameter: Number, symbols_map: Dict[sympy.Symbol, Parameter]) -> Number:
    return parameter


@_sub_symbols.register
def _sub_symbols_in_expression(parameter: sympy.Expr, symbols_map: Dict[sympy.Symbol, Parameter]) -> sympy.Expr:
    return parameter.subs(symbols_map)


@_sub_symbols.register
def _sub_symbols_in_symbol(parameter: sympy.Symbol, symbols_map: Dict[sympy.Symbol, Parameter]) -> Parameter:
    return symbols_map.get(parameter, parameter)


@dataclass(frozen=True)
class MatrixFactoryGate:
    """`Gate` protocol implementation with a deferred matrix construction.

    Most built-in gates are instances of this class.

    Keeping a `matrix_factory` instead of a plain gate matrix allows us to defer matrix
    construction to _after_ parameter binding. This saves unnecessary work in scenarios
    where we construct a quantum circuit and immediately bind parameter values. When done
    multiple times, e.g. for every gate in each optimization step, this can lead to major
    performance issues.

    Args:
        name: Name of this gate. Implementers of new gates should make sure that the names are
            unique.
        matrix_factory: a callable mapping arbitrary number of parameters into gate matrix.
            Implementers of new gates should make sure the returned matrices are
            square and of dimension being 2 ** `num_qubits`.
        params: params bound to this instance of gate. Actual matrix of this gate will be
            constructed, upon request, by passing params to `matrix_factory`.
        num_qubits: number of qubits this gate acts on.
    """

    name: str
    matrix_factory: Callable[..., sympy.Matrix]
    params: Tuple[Parameter, ...]
    num_qubits: int
    is_hermitian: bool = False

    @property
    def matrix(self) -> sympy.Matrix:
        """Unitary matrix defining action of this gate.

        This is a computed property using `self.matrix_factory` called
        with parameters bound to this gate.
        """
        return self.matrix_factory(*self.params)

    def bind(self, symbols_map) -> "MatrixFactoryGate":
        new_symbols = tuple(_sub_symbols(param, symbols_map) for param in self.params)
        return MatrixFactoryGate(
            name=self.name,
            matrix_factory=self.matrix_factory,
            params=new_symbols,
            num_qubits=self.num_qubits
        )

    def controlled(self, num_controlled_qubits: int) -> Gate:
        return ControlledGate(self, num_controlled_qubits)

    @property
    def dagger(self) -> Gate:
        return self if self.is_hermitian else Dagger(self)

    def __str__(self):
        return (
            f"{self.name}({', '.join(map(str,self.params))})"
            if self.params
            else self.name
        )

    __call__ = Gate.__call__


@dataclass(frozen=True)
class ControlledGate(Gate):
    wrapped_gate: Gate
    num_control_qubits: int

    @property
    def name(self):
        return "control"

    @property
    def num_qubits(self):
        return self.wrapped_gate.num_qubits + self.num_control_qubits

    @property
    def matrix(self):
        return sympy.Matrix.diag(
            sympy.eye(2 ** self.num_qubits - 2 ** self.wrapped_gate.num_qubits),
            self.wrapped_gate.matrix
        )

    @property
    def params(self):
        return self.wrapped_gate.params

    def controlled(self, num_control_qubits: int) -> "ControlledGate":
        return ControlledGate(
            wrapped_gate=self.wrapped_gate,
            num_control_qubits=self.num_control_qubits + num_control_qubits
        )

    @property
    def dagger(self) -> "ControlledGate":
        return ControlledGate(
            wrapped_gate=self.wrapped_gate.dagger,
            num_control_qubits=self.num_control_qubits
        )

    def bind(self, symbols_map) -> "Gate":
        return self.wrapped_gate.bind(symbols_map).controlled(self.num_control_qubits)


@dataclass(frozen=True)
class Dagger(Gate):
    wrapped_gate: Gate

    @property
    def matrix(self) -> sympy.Matrix:
        return self.wrapped_gate.matrix.adjoint()

    @property
    def params(self) -> Tuple[Parameter, ...]:
        return self.wrapped_gate.params

    @property
    def num_qubits(self) -> int:
        return self.wrapped_gate.num_qubits

    @property
    def name(self):
        return "dagger"

    def controlled(self, num_control_qubits: int) -> Gate:
        return self.wrapped_gate.controlled(num_control_qubits).dagger

    def bind(self, symbols_map) -> "Gate":
        return self.wrapped_gate.bind(symbols_map).dagger

    @property
    def dagger(self) -> "Gate":
        return self.wrapped_gate


def _matrix_substitution_func(matrix: sympy.Matrix, symbols):
    """Create a function that substitutes value for free params to given matrix.

    This is meant to be used as a factory function in CustomGates, where
    one already has a matrix.

    Args:
        matrix: a matrix with symbolic parameters.
        symbols: an iterable comprising all symbolic (free) params of matrix.
    Returns:
        A callable f such that f(param_1, ..., param_n) returns matrix resulting
        from substituting free symbols in `matrix` with param_1,...,param_n
        in the order specified by `symbols`.
    """

    def _substitution_func(*params):
        return matrix.subs({symbol: arg for symbol, arg in zip(symbols, params)})

    return _substitution_func


def define_gate_with_matrix(
    name: str, matrix: sympy.Matrix, symbols_ordering: Tuple[sympy.Symbol, ...]
) -> Callable[..., MatrixFactoryGate]:
    """Makes it easy to define custom gates.

    Define new gate specified by a (possibly parametrized) matrix.

    Note that this is slightly less efficient, but more convenient, than creating
    a callable that returns a matrix and passing it to CustomGate.

    Args:
        name: name of the gate.
        matrix: matrix of the gate. It should a matrix of shape 2 ** N x 2 ** N,
            where N is the number of qubits this gate acts on.
        symbols_ordering: tuple defining order in which symbols should be passed to the gates
            initializer.
            For instance:
            >>>U = define_gate_with_matrix(
                "U",
                Matrix([
                    [Symbol("a"), 0],
                    [0, Symbol("b")]
                ]),
                (Symbol("a"), Symbol("b"))
            )
            then, when runniing:
            >>>V = U(-0.5, 0.7)
            V will be defined by substituting a=-0.5 and b=0.7,
            resulting in a gate V identical to
            >>>V = MatrixFactoryGate(
                "U",
                lambda a, b: Matrix([
                    [a, 0],
                    [0, b]
                ]),
                (-0.5, 0.7)
            )

    Returns:
        Callable mapping parameters into an instance of the defined gate.
    """
    n_qubits = math.floor(math.log2(matrix.shape[0]))
    if 2 ** n_qubits != matrix.shape[0] or 2 ** n_qubits != matrix.shape[1]:
        raise ValueError("Gate's matrix has to be square with dimension 2^N")

    def _gate(*params):
        return MatrixFactoryGate(
            name, _matrix_substitution_func(matrix, symbols_ordering), params, n_qubits
        )

    return _gate


def _circuit_size_by_operations(operations):
    return (
        0
        if not operations
        else max(qubit_index for operation in operations for qubit_index in operation.qubit_indices) + 1
    )


class Circuit:
    """ZQuantum representation of a quantum circuit."""
    def __init__(self, operations: Optional[Iterable[GateOperation]] = None, n_qubits: Optional[int] = None):
        self._operations = list(operations) if operations is not None else []
        self._n_qubits = (
            n_qubits if n_qubits is not None else _circuit_size_by_operations(self._operations)
        )

    @property
    def operations(self):
        """Sequence of quantum gates to apply to qubits in this circuit."""
        return self._operations

    @property
    def n_qubits(self):
        """Number of qubits in this circuit.
        Not every qubit has to be used by a gate.
        """
        return self._n_qubits

    @property
    def symbolic_params(self):
        """Set of all the sympy symbols used as params of gates in the circuit."""
        return reduce(set.union, (set(gate.symbolic_params) for gate in self._operations), set())

    def __eq__(self, other: "Circuit"):
        if not isinstance(other, type(self)):
            return False

        if self.n_qubits != other.n_qubits:
            return False

        if list(self.operations) != list(other.operations):
            return False

        return True

    def __add__(self, other: Union["Circuit"]):
        return _append_to_circuit(other, self)

    def bind(self, symbols_map: Dict[sympy.Symbol, Any]):
        """Create a copy of the current Circuit with the parameters of each gate evaluated to the values
        provided in the input symbols map

        Args:
            symbols_map (Dict): A map of the symbols/gate parameters to new values
        """
        raise NotImplementedError()

    def to_dict(self):
        """Creates a dictionary representing a circuit.
        The dictionary is serializable to JSON.

        Returns:
            A mapping with keys:
                - "schema"
                - "n_qubits"
                - "symbolic_params"
                - "gates"
        """
        raise NotImplementedError()

    @classmethod
    def from_dict(cls, json_dict):
        raise NotImplementedError()


@singledispatch
def _append_to_circuit(other, circuit: Circuit):
    raise NotImplementedError()


@_append_to_circuit.register
def _append_operation(other: GateOperation, circuit: Circuit):
    n_qubits_by_operation = max(other.qubit_indices) + 1
    return type(circuit)(
        operations=[*circuit.operations, other],
        n_qubits=max(circuit.n_qubits, n_qubits_by_operation),
    )


@_append_to_circuit.register
def _append_circuit(other: Circuit, circuit: Circuit):
    return type(circuit)(
        operations=[*circuit.operations, *other.operations],
        n_qubits=max(circuit.n_qubits, other.n_qubits),
    )
