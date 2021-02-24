"""Class hierarchy for base gates."""
import math
from dataclasses import dataclass, field
from numbers import Number
from typing import Tuple, Union, Callable, Optional

import sympy
from typing_extensions import Protocol

Parameter = Union[sympy.Symbol, Number]


class Gate(Protocol):
    """Quantum gate."""

    @property
    def name(self) -> str:
        """Name of the gate.

        Name is used in textual representation and dispatching in conversion between
        frameworks, therefore implementers of this protocol should make sure
        the default gate names are not used.
        """
        raise NotImplementedError()

    @property
    def params(self) -> Tuple[Parameter]:
        """Value of parameters bound to this gate.

        Length of `params` should be equal to number of parameters in gate's initializer.
        In particular, nonparametric gates should always return ().
        """
        raise NotImplementedError()

    @property
    def num_qubits(self) -> int:
        """Number of qubits this gate acts on."""
        raise NotImplementedError()

    @property
    def matrix(self) -> sympy.Matrix:
        """Unitary matrix describing gate's action on state vector."""
        raise NotImplementedError()


@dataclass(frozen=True)
class CustomGate:
    """Custom gate defined using matrix factory.

    Args:
        name: Name of this gate. Implementers of new gates should make sure that the names are
            unique.
        matrix_factory: a callable mapping arbitrary number of parameters into gate matrix.
            Implementers of new gates should make sure the returned matrices are
            square and of dimension being 2 ** `num_qubits`.
        params: params boumd to this instance of gate. Actual matrix of this gate will be
            constructed, upon request, by passing params to `matrix_factory`.
        num_qubits: number of qubits this gate acts on.
    """

    name: str
    matrix_factory: Callable[..., sympy.Matrix]
    params: Tuple[Parameter, ...]
    num_qubits: int
    _matrix: Optional[sympy.Matrix] = field(init=False, default=None)

    @property
    def matrix(self) -> sympy.Matrix:
        """Unitary matrix defining action of this gate.

        This is a cached property computed using `self.matrix_factory` called
        with parameters bound to this gate.
        """
        if self._matrix is None:
            # object.__setattr__ is used because directly setting attribute on instances
            # of frozen dataclass is prohibited.
            object.__setattr__(self, "_matrix", self.matrix_factory(*self.params))
        return self._matrix

    def __repr__(self):
        return (
            f"{self.name}({', '.join(map(str,self.params))})"
            if self.params
            else self.name
        )


def _matrix_substitution_func(matrix: sympy.Matrix, symbols):
    """Create a function that substitutes value for free params to given matrix.

    This is meant to be used as a factory function in CustomGates, where
    one already has a matrix.

    Args:
        matrix: a matrix with symbolic parameters.
        symbols: an iterable comprising all symbolic (free) params of matrix.
    Returns:
        A callable f such that f(param1, ..., paramn) returns matrix resulting
        from substituting free symbols in `matrix` with param1,...,paramn
        in the order specified by `symbols`.
    """

    def _substitution_func(*values):
        return matrix.subs({symbol: value for symbol, value in zip(symbols, values)})

    return _substitution_func


def _n_qubits_for_matrix(matrix_shape):
    n_qubits = math.floor(math.log2(matrix_shape[0]))
    if 2 ** n_qubits != matrix_shape[0] or 2 ** n_qubits != matrix_shape[1]:
        raise ValueError("Gate's matrix has to be square with dimension 2^N")

    return n_qubits


def define_gate(
    name: str, matrix: sympy.Matrix, free_symbols: Tuple[sympy.Symbol, ...]
) -> Callable[..., CustomGate]:
    """Define new gate specified by a (possibly parametrized) matrix.

    Consider it a helper that handles constructing a `matrix_factory` for CustomGate init.

    Args:
        name: name of the gate.
        matrix: matrix of the gate. It should a matrix of shape 2 ** N x 2 ** N,
            where N is the number of qubits this gate acts on.
        free_symbols: tuple defining order in which symbols should be passed to the gates
            initializer.
            For instances, if U = define_gate("U", some_matrix, (Symbol("a"), Symbol("b")))
            then matrix of U(1, 2) will be defined by substitute a=1 and b=2
            into some_matrix.
    Returns:
        Callable mapping parameters into an instance of the defined gate.
    """
    # n_qubits = math.floor(math.log2(matrix.shape[0]))
    # if 2 ** n_qubits != matrix.shape[0] or 2 ** n_qubits != matrix.shape[1]:
    #     raise ValueError("Gate's matrix has to be square with dimension 2^N")

    n_qubits = _n_qubits_for_matrix(matrix.shape)

    def _gate_factory(*params):
        return CustomGate(
            name=name,
            matrix_factory=_matrix_substitution_func(matrix, free_symbols),
            params=params,
            num_qubits=n_qubits
        )

    return _gate_factory


def define_nonparametric_gate(
    name: str,
    matrix: sympy.Matrix
):
    n_qubits = _n_qubits_for_matrix(matrix.shape)

    def _gate_factory():
        return CustomGate(
            name=name,
            matrix_factory=lambda: matrix,
            params=(),
            num_qubits=n_qubits
        )

    return _gate_factory


def define_one_param_gate(
    name: str,
    matrix_factory,
    n_qubits
):
    def _gate_factory(param):
        return CustomGate(
            name=name,
            matrix_factory=matrix_factory,
            params=(param,),
            num_qubits=n_qubits
        )

    return _gate_factory
