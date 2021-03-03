"""Class hierarchy for base gates."""
import math
from dataclasses import dataclass
from functools import singledispatch, reduce
from numbers import Number
from typing import Tuple, Union, Callable, Dict, Optional, Iterable, Any, List

import sympy
from typing_extensions import Protocol

from ...utils import SCHEMA_VERSION
from . import _builtin_gates

Parameter = Union[sympy.Symbol, Number]


def _jsonify_param(param: Parameter):
    return str(param)


class Gate(Protocol):
    """Quantum gate representable by a matrix, translatable to other frameworks
    and backends."""

    @property
    def name(self) -> str:
        """Globally unique name of the gate.

        Name is used in textual representation and dispatching in conversion between
        frameworks. Defining different gates with the same name as built-in ones
        is discouraged."""
        raise NotImplementedError()

    @property
    def params(self) -> Tuple[Parameter, ...]:
        """Value of parameters bound to this gate.

        Length of `params` should be equal to number of parameters in gate's initializer.
        In particular, nonparametric gates should always return ().

        We need it for translations to other frameworks and for serialization.
        """
        raise NotImplementedError()

    @property
    def free_symbols(self):
        """Unbound symbols.

        Number of free symbols is greater or equal to number of params - you can use
        a single Sympy expression with multiple symbols as a single param.
        """
        symbols = set(
            symbol
            for param in self.params
            if isinstance(param, sympy.Expr)
            for symbol in param.free_symbols
        )
        return sorted(symbols, key=str)

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

    def to_dict(self):
        return {
            "name": self.name,
            **(
                {"params": list(map(_jsonify_param, self.params))}
                if self.params
                else {}
            ),
            **(
                {"free_symbols": sorted(map(str, self.free_symbols))}
                if self.free_symbols
                else {}
            ),
        }


def _gate_from_dict(dict_, custom_gate_defs):
    """Prototype implementation of circuit deserialization"""
    gate_ref = _builtin_gates.builtin_gate_by_name(dict_["name"])
    if gate_ref is not None:
        # ATM we don't have a better way to check if the serialized gate was parametric
        # or not
        if isinstance(gate_ref, MatrixFactoryGate):
            return gate_ref
        else:
            symbols_map = _make_symbols_map(dict_.get("free_symbols", []))
            return gate_ref(
                *[_deserialize_term(param, symbols_map) for param in dict_["params"]]
            )

    if dict_["name"] == ControlledGate.__name__:
        raise NotImplementedError()

    if dict_["name"] == Dagger.__name__:
        raise NotImplementedError()

    gate_def = next(
        (
            gate_def
            for gate_def in custom_gate_defs
            if gate_def.gate_name == dict_["name"]
        ),
        None,
    )
    if gate_def is None:
        raise ValueError(
            f"Custom gate definition for {dict_['name']} missing from serialized dict"
        )

    symbols_map = _make_symbols_map(map(str, gate_def.params_ordering))
    return gate_def(
        *[_deserialize_term(param, symbols_map) for param in dict_["params"]]
    )
    # TODO:
    # - controlled gate
    # - dagger


@dataclass(frozen=True)
class GateOperation:
    gate: Gate
    qubit_indices: Tuple[int, ...]

    def to_dict(self):
        return {
            "type": "gate_operation",
            "gate": self.gate.to_dict(),
            "qubit_indices": list(self.qubit_indices),
        }

    @classmethod
    def from_dict(cls, dict_, custom_gate_defs):
        return cls(
            gate=_gate_from_dict(dict_["gate"], custom_gate_defs),
            qubit_indices=tuple(dict_["qubit_indices"]),
        )

    def __str__(self):
        return f"{self.gate}({','.join(map(str, self.qubit_indices))})"


GATE_OPERATION_DESERIALIZERS = {"gate_operation": GateOperation.from_dict}


def _gate_operation_from_dict(dict_, custom_gate_defs):
    # Add deserializers here when we need to support custom, non-gate operations
    return GATE_OPERATION_DESERIALIZERS[dict_["type"]](dict_, custom_gate_defs)


@singledispatch
def _sub_symbols(parameter, symbols_map: Dict[sympy.Symbol, Parameter]) -> Parameter:
    raise NotImplementedError()


@_sub_symbols.register
def _sub_symbols_in_number(
    parameter: Number, symbols_map: Dict[sympy.Symbol, Parameter]
) -> Number:
    return parameter


@_sub_symbols.register
def _sub_symbols_in_expression(
    parameter: sympy.Expr, symbols_map: Dict[sympy.Symbol, Parameter]
) -> sympy.Expr:
    return parameter.subs(symbols_map)


@_sub_symbols.register
def _sub_symbols_in_symbol(
    parameter: sympy.Symbol, symbols_map: Dict[sympy.Symbol, Parameter]
) -> Parameter:
    return symbols_map.get(parameter, parameter)


@dataclass(frozen=True)
class MatrixFactoryGate:
    """`Gate` protocol implementation with a deferred matrix construction.

    Most built-in gates are instances of this class.
    It requires the gate definition to be present during deserialization, so it's not
    easily applicable for gates defined in Orquestra steps.

    Keeping a `matrix_factory` instead of a plain gate matrix allows us to defer matrix
    construction to _after_ parameter binding. This saves unnecessary work in scenarios
    where we construct a quantum circuit and immediately bind parameter values. When done
    multiple times, e.g. for every gate in each optimization step, this can lead to major
    performance issues.

    Args:
        name: Name of this gate. Implementers of new gates should make sure that the names
            are unique.
        matrix_factory: a callable mapping arbitrary number of parameters into gate
            matrix. Implementers of new gates should make sure the returned matrices are
            square and of dimension being 2 ** `num_qubits`.
        params: gate parameters - either concrete values or opaque symbols.
            Will be passed to `matrix_factory` when `matrix` property is requested.
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
            num_qubits=self.num_qubits,
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

    # Normally, we'd use the default implementations by inheriting from the Gate protocol.
    # We can't do that because of __init__ arg default value issues, this is
    # the workaround.
    free_symbols = Gate.free_symbols
    __call__ = Gate.__call__
    to_dict = Gate.to_dict


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
            self.wrapped_gate.matrix,
        )

    @property
    def params(self):
        return self.wrapped_gate.params

    def controlled(self, num_control_qubits: int) -> "ControlledGate":
        return ControlledGate(
            wrapped_gate=self.wrapped_gate,
            num_control_qubits=self.num_control_qubits + num_control_qubits,
        )

    @property
    def dagger(self) -> "ControlledGate":
        return ControlledGate(
            wrapped_gate=self.wrapped_gate.dagger,
            num_control_qubits=self.num_control_qubits,
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


def _n_qubits(matrix):
    n_qubits = math.floor(math.log2(matrix.shape[0]))
    if 2 ** n_qubits != matrix.shape[0] or 2 ** n_qubits != matrix.shape[1]:
        raise ValueError("Gate's matrix has to be square with dimension 2^N")
    return n_qubits


def _matrix_to_json(matrix: sympy.Matrix):
    return [
        [str(element) for element in matrix.row(row_i)]
        for row_i in range(matrix.shape[0])
    ]


def _make_symbols_map(symbol_names):
    return {name: sympy.Symbol(name) for name in symbol_names}


def _deserialize_term(term: Union[str, Number], symbols_map: Dict[str, sympy.Symbol]):
    # We pass symbols_map because some commonly used symbol names (e.g. gamma) are
    # by default parsed as functions from sympy instead of symbols.
    return sympy.sympify(term, locals=symbols_map) if isinstance(term, str) else term


def _matrix_from_json(
    json_rows: List[List[str]], symbols_names: Iterable[str]
) -> sympy.Matrix:
    symbols_map = _make_symbols_map(symbols_names)
    return sympy.Matrix(
        [
            [_deserialize_term(element, symbols_map) for element in json_row]
            for json_row in json_rows
        ]
    )


@dataclass(frozen=True)
class FixedMatrixFactory:
    matrix: sympy.Matrix
    params_ordering: Tuple[Parameter, ...]

    def __call__(self, *gate_params):
        return self.matrix.subs({symbol: arg for symbol, arg in zip(self.params_ordering, gate_params)})


@dataclass(frozen=True)
class CustomGateDefinition:
    gate_name: str
    matrix: sympy.Matrix
    params_ordering: Tuple[sympy.Symbol, ...]

    def __post_init__(self):
        n_qubits = _n_qubits(self.matrix)
        object.__setattr__(self, "_n_qubits", n_qubits)

    def __call__(self, *params):
        return MatrixFactoryGate(
            self.gate_name,
            FixedMatrixFactory(self.matrix, self.params_ordering),
            params,
            self._n_qubits,
        )

    def to_dict(self):
        return {
            "gate_name": self.gate_name,
            "matrix": _matrix_to_json(self.matrix),
            "params_ordering": list(map(_jsonify_param, self.params_ordering)),
        }

    @classmethod
    def from_dict(cls, dict_):
        symbols = [sympy.Symbol(term) for term in dict_.get("params_ordering", [])]
        return cls(
            gate_name=dict_["gate_name"],
            matrix=_matrix_from_json(dict_["matrix"], dict_.get("params_ordering", [])),
            params_ordering=tuple(symbols),
        )


def _circuit_size_by_operations(operations):
    return (
        0
        if not operations
        else max(
            qubit_index
            for operation in operations
            for qubit_index in operation.qubit_indices
        )
        + 1
    )


def _bind_operation(op: GateOperation, symbols_map) -> GateOperation:
    return op.gate.bind(symbols_map)(*op.qubit_indices)


CIRCUIT_SCHEMA = SCHEMA_VERSION + "-circuit"


class Circuit:
    """ZQuantum representation of a quantum circuit."""

    def __init__(
        self,
        operations: Optional[Iterable[GateOperation]] = None,
        n_qubits: Optional[int] = None,
        custom_gate_definitions: Optional[Iterable[CustomGateDefinition]] = None,
    ):
        self._operations = list(operations) if operations is not None else []
        self._n_qubits = (
            n_qubits
            if n_qubits is not None
            else _circuit_size_by_operations(self._operations)
        )
        self._custom_gate_definitions = (
            list(custom_gate_definitions) if custom_gate_definitions else []
        )

    @property
    def operations(self):
        """Sequence of quantum gates to apply to qubits in this circuit."""
        return self._operations

    @property
    def custom_gate_definitions(self):
        return self._custom_gate_definitions

    @property
    def n_qubits(self):
        """Number of qubits in this circuit.
        Not every qubit has to be used by a gate.
        """
        return self._n_qubits

    @property
    def free_symbols(self):
        """Set of all the sympy symbols used as params of gates in the circuit."""
        return reduce(
            set.union,
            (operation.gate.free_symbols for operation in self._operations),
            set(),
        )

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
        """Create a copy of the current circuit with the parameters of each gate bound to
        the values provided in the input symbols map

        Args:
            symbols_map: A map of the symbols/gate parameters to new values
        """
        return type(self)(
            operations=[_bind_operation(op, symbols_map) for op in self.operations],
            n_qubits=self.n_qubits,
        )

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
        return {
            "schema": CIRCUIT_SCHEMA,
            "n_qubits": self.n_qubits,
            **(
                {
                    "operations": [
                        operation.to_dict() for operation in self.operations
                    ],
                }
                if self.operations
                else {}
            ),
            **(
                {
                    "custom_gate_definitions": [
                        gate_def.to_dict() for gate_def in self.custom_gate_definitions
                    ]
                }
                if self.custom_gate_definitions
                else {}
            ),
        }

    @classmethod
    def from_dict(cls, dict_):
        defs = [
            CustomGateDefinition.from_dict(def_dict)
            for def_dict in dict_.get("custom_gate_definitions", [])
        ]
        return cls(
            operations=[
                _gate_operation_from_dict(op_dict, defs)
                for op_dict in dict_.get("operations", [])
            ],
            n_qubits=dict_["n_qubits"],
            custom_gate_definitions=defs,
        )

    def __repr__(self):
        return f"{type(self).__name__}(operations=[{', '.join(map(str, self.operations))}], n_qubits={self.n_qubits}, custom_gate_definitions={self.custom_gate_definitions})"


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
