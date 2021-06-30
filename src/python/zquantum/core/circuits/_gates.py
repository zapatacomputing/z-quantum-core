"""Data structures for ZQuantum gates."""
import math
from dataclasses import dataclass, replace
from typing import Callable, Dict, Iterable, Sequence, Tuple, Union

import numpy as np
import sympy
from typing_extensions import Protocol, runtime_checkable

from ._operations import Parameter, get_free_symbols, sub_symbols
from ._unitary_tools import _lift_matrix_numpy, _lift_matrix_sympy


@runtime_checkable
class Gate(Protocol):
    """Interface of a quantum gate representable by a matrix, translatable to other
    frameworks and backends.

    See `zquantum.core.circuits` for a list of built-in gates and usage guide.
    """

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

        Length of `params` should be equal to number of parameters in gate initializer.
        In particular, nonparametric gates should always return ().

        Examples:
        - an `H` gate has no params
        - a `RX(np.pi)` gate has a single param with value of `np.pi`
        - a `RX(sympy.Symbol("theta"))` gate has a single symbolic param `theta`
        - a `RX(sympy.sympify("theta * alpha"))` gate has a single symbolic expression
            param `theta*alpha`

        We need it for translations to other frameworks and for serialization.
        """
        raise NotImplementedError()

    @property
    def free_symbols(self) -> Iterable[sympy.Symbol]:
        """Unbound symbols in the gate matrix.

        Examples:
        - an `H` gate has no free symbols
        - a `RX(np.pi)` gate has no free symbols
        - a `RX(sympy.Symbol("theta"))` gate has a single free symbol `theta`
        - a `RX(sympy.sympify("theta * alpha"))` gate has two free symbols, `alpha` and
            `theta`
        - a `RX(sympy.sympify("theta * alpha")).bind({sympy.Symbol("theta"): 0.42})`
            gate has one free symbol, `alpha`
        """
        return get_free_symbols(self.params)

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

    def replace_params(self, new_params: Tuple[Parameter, ...]) -> "Gate":
        raise NotImplementedError()

    def __call__(self, *qubit_indices: int) -> "GateOperation":
        """Returns representation of applying this gate on qubits in a circuit."""
        return GateOperation(self, qubit_indices)


def gate_is_parametric(gate_ref, gate_params):
    return not not gate_params


@dataclass(frozen=True)
class GateOperation:
    """Represents applying a `Gate` to 1 or more qubits in a circuit."""

    gate: Gate
    qubit_indices: Tuple[int, ...]

    @property
    def params(self) -> Tuple[Parameter, ...]:
        return self.gate.params

    def bind(self, symbols_map: Dict[sympy.Symbol, Parameter]) -> "GateOperation":
        return GateOperation(self.gate.bind(symbols_map), self.qubit_indices)

    def replace_params(self, new_params: Tuple[Parameter, ...]) -> "GateOperation":
        return GateOperation(self.gate.replace_params(new_params), self.qubit_indices)

    def lifted_matrix(self, num_qubits):
        return (
            _lift_matrix_sympy(self.gate.matrix, self.qubit_indices, num_qubits)
            if self.gate.free_symbols
            else _lift_matrix_numpy(self.gate.matrix, self.qubit_indices, num_qubits)
        )

    def apply(self, wavefunction: Sequence[Parameter]) -> Sequence[Parameter]:
        num_qubits = np.log2(len(wavefunction))
        if 2 ** num_qubits != len(wavefunction):
            raise ValueError(
                "GateOperation can only be applied to multi-qubit state vector but "
                f"vector of length {len(wavefunction)} was provided."
            )

        return self.lifted_matrix(int(num_qubits)) @ wavefunction

    @property
    def free_symbols(self) -> Iterable[sympy.Symbol]:
        return self.gate.free_symbols

    def __str__(self):
        return f"{self.gate}({','.join(map(str, self.qubit_indices))})"


def _all_attrs_equal(obj, other_obj, attrs):
    return all(getattr(obj, attr) == getattr(other_obj, attr) for attr in attrs)


@dataclass(frozen=True)
class MatrixFactoryGate:
    """Data structure for a `Gate` with deferred matrix construction.

    Most built-in gates are instances of this class.
    See `zquantum.core.circuits` for built-in gates and usage guide.

    This class requires the gate definition to be present during deserialization, so
    it's not easily applicable for gates defined in Orquestra steps. If you want to
    define a new gate, check out `CustomGateDefinition` first.

    Keeping a `matrix_factory` instead of a plain gate matrix allows us to defer matrix
    construction to _after_ parameter binding. This saves unnecessary work in scenarios
    where we construct a quantum circuit and immediately bind parameter values. When
    done multiple times, e.g. for every gate in each optimization step, this can lead
    to major performance issues.

    Args:
        name: Name of this gate. Implementers of new gates should make sure that the
            names are unique.
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

        This is a computed property using `self.matrix_factory` called with parameters
            bound to this gate.
        """
        return self.matrix_factory(*self.params)

    def bind(self, symbols_map) -> "MatrixFactoryGate":
        return self.replace_params(
            tuple(sub_symbols(param, symbols_map) for param in self.params)
        )

    def replace_params(self, new_params: Tuple[Parameter, ...]) -> "MatrixFactoryGate":
        return replace(self, params=new_params)

    def controlled(self, num_controlled_qubits: int) -> Gate:
        return ControlledGate(self, num_controlled_qubits)

    @property
    def dagger(self) -> Union["MatrixFactoryGate", Gate]:
        return self if self.is_hermitian else Dagger(self)

    def __str__(self):
        return (
            f"{self.name}({', '.join(map(str,self.params))})"
            if self.params
            else self.name
        )

    def __eq__(self, other):
        if type(self) != type(other):
            return False

        if not _all_attrs_equal(
            self, other, set(self.__dataclass_fields__) - {"params"}
        ):
            return False

        if len(self.params) != len(other.params):
            return False

        return all(
            _are_matrix_elements_equal(p1, p2)
            for p1, p2 in zip(self.params, other.params)
        )

    # Normally, we'd use the default implementations by inheriting from the Gate
    # protocol.  We can't do that because of __init__ arg default value issues, this is
    # the workaround.
    @property
    def free_symbols(self) -> Iterable[sympy.Symbol]:
        """Unbound symbols in the gate matrix. See Gate.free_symbols for details."""
        return get_free_symbols(self.params)

    __call__ = Gate.__call__


CONTROLLED_GATE_NAME = "Control"


@dataclass(frozen=True)
class ControlledGate(Gate):
    wrapped_gate: Gate
    num_control_qubits: int

    @property
    def name(self):
        return CONTROLLED_GATE_NAME

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

    def replace_params(self, new_params: Tuple[Parameter, ...]) -> "Gate":
        return self.wrapped_gate.replace_params(new_params).controlled(
            self.num_control_qubits
        )


DAGGER_GATE_NAME = "Dagger"


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
        return DAGGER_GATE_NAME

    def controlled(self, num_control_qubits: int) -> Gate:
        return self.wrapped_gate.controlled(num_control_qubits).dagger

    def bind(self, symbols_map) -> "Gate":
        return self.wrapped_gate.bind(symbols_map).dagger

    def replace_params(self, new_params: Tuple[Parameter, ...]) -> "Gate":
        return self.wrapped_gate.replace_params(new_params).dagger

    @property
    def dagger(self) -> "Gate":
        return self.wrapped_gate


def _n_qubits(matrix):
    n_qubits = math.floor(math.log2(matrix.shape[0]))
    if 2 ** n_qubits != matrix.shape[0] or 2 ** n_qubits != matrix.shape[1]:
        raise ValueError("Gate's matrix has to be square with dimension 2^N")
    return n_qubits


@dataclass(frozen=True)
class CustomGateMatrixFactory:
    """Can be passed as `matrix_factory` when a gate matrix isn't lazily evaluated."""

    gate_definition: "CustomGateDefinition"

    @property
    def matrix(self) -> sympy.Matrix:
        return self.gate_definition.matrix

    @property
    def params_ordering(self) -> Tuple[Parameter, ...]:
        return self.gate_definition.params_ordering

    def __call__(self, *gate_params):
        return self.matrix.subs(
            {symbol: arg for symbol, arg in zip(self.params_ordering, gate_params)}
        )

    def __eq__(self, other):
        if type(self) != type(other):
            return False

        if self.params_ordering != other.params_ordering:
            return False

        if not _are_matrices_equal(self.matrix, other.matrix):
            return False

        return True


@dataclass(frozen=True)
class CustomGateDefinition:
    """Use this class to define a non-built-in gate.

    See "Defining new gates" section in `help(zquantum.core.circuits)` for
    usage guide.

    User-defined gates are treated differently than the built-in ones,
    because the built-in ones are defined in `zquantum.core` library, and so
    we can assume that the definitions will be available during circuit deserialization.

    User-provided gates can be defined in one repo (e.g. Orquestra step), serialized,
    and passed to another project for deserialization. The other project must have
    access to gate details, e.g. the gate matrix. This class is designed to keep track
    of the gate details needed for deserialization.

    Instances of this class are serialized by the Circuit objects, additionally to
    Circuit operations.
    """

    gate_name: str
    matrix: sympy.Matrix
    params_ordering: Tuple[sympy.Symbol, ...]

    def __post_init__(self):
        n_qubits = _n_qubits(self.matrix)
        object.__setattr__(self, "_n_qubits", n_qubits)

    def __call__(self, *gate_params):
        return MatrixFactoryGate(
            self.gate_name,
            CustomGateMatrixFactory(self),
            gate_params,
            self._n_qubits,
        )

    def __eq__(self, other):
        if type(self) != type(other):
            return False

        if self.gate_name != other.gate_name:
            return False

        if self.params_ordering != other.params_ordering:
            return False

        if not _are_matrices_equal(self.matrix, other.matrix):
            return False

        return True


def _are_matrix_elements_equal(element, another_element):
    """Determine if two elements from gates' matrices are equal.

    This is to be used in __eq__ method when comparing matrices elementwise.

    Args:
        element: first value to compare. It can be float, complex or a sympy expression.
        another_element: second value to compare.
    """
    difference = sympy.N(sympy.expand(element) - sympy.expand(another_element))

    try:
        return np.allclose(
            float(sympy.re(difference)) + 1j * float(sympy.im(difference)), 0
        )
    except TypeError:
        return False


def _are_matrices_equal(matrix, another_matrix):
    return all(
        _are_matrix_elements_equal(element, another_element)
        for element, another_element in zip(matrix, another_matrix)
    )
