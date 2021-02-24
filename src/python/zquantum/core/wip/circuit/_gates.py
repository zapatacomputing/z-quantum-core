"""Class hierarchy for base gates."""
from numbers import Number
from typing import Tuple, Union

import sympy
from typing_extensions import Protocol

Parameter = Union[sympy.Symbol, Number]


class Gate(Protocol):

    @property
    def name(self) -> str:
        raise NotImplementedError()

    @property
    def params(self) -> Tuple[Parameter]:
        raise NotImplementedError()

    @property
    def num_qubits(self) -> int:
        raise NotImplementedError()

    @property
    def matrix(self) -> sympy.Matrix:
        raise NotImplementedError()
