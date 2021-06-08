from numbers import Number
from typing import Protocol, Tuple, Union, Dict, TypeVar

import sympy

Parameter = Union[sympy.Symbol, Number]

T = TypeVar("T", bound="Operation")


class Operation(Protocol):

    @property
    def params(self) -> Tuple[Parameter, ...]:
        """Parameters of this operation."""
        raise NotImplementedError()

    def bind(self: T, symbols_map: Dict[sympy.Symbol, Parameter]) -> T:
        """Create new operation by replacing free symbols in operation params.

        The operation returned by this method should be of the same type
        as self, e.g. binding parameters to GateOperation should produce
        GateOperation.
        """
        raise NotImplementedError()

    def replace_params(self: T, new_params: Tuple[Parameter, ...]) -> T:
        """Create new operation by replacing params.

        The difference between bind and replace params is that it bind performs
        parameter substitution - in particular, parameters without free symbols
        are unaffected by bind, whereas replace_params replaces *all* params.
        """
        raise NotImplementedError()

    @property
    def free_symbols(self):
        """Free symbols parameterizing this operation.

        Note that number of free_symbols is unrelated to number of params.
        Some params can be expressions with multiple free symbols, while other params
        might not comprise free symbols at all.
        """
        raise NotImplementedError()
