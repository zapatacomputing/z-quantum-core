from numbers import Number
from typing import Protocol, Tuple, Union, Dict, TypeVar

import sympy

Parameter = Union[sympy.Symbol, Number]

T = TypeVar("T", bound="Operation")


class Operation(Protocol):

    @property
    def params(self) -> Tuple[Parameter, ...]:
        raise NotImplementedError()

    def bind(self: T, symbols_map: Dict[sympy.Symbol, Parameter]) -> T:
        raise NotImplementedError()

    def replace_params(self: T, new_params: Tuple[Parameter, ...]) -> T:
        raise NotImplementedError()

    def free_symbols(self):
        raise NotImplementedError()
