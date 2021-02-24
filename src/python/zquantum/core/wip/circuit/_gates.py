"""Class hierarchy for base gates."""
from numbers import Number
from typing import Tuple, Union

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
