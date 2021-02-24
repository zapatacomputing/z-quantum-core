"""Class hierarchy for base gates."""
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
            f"{self.name}({', '.join(map(str,self.params))})" if self.params
            else self.name
        )
