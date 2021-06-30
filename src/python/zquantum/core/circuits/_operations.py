from abc import abstractmethod
from functools import singledispatch
from numbers import Number
from typing import Dict, Iterable, Sequence, Tuple, TypeVar, Union

import sympy
from typing_extensions import Protocol

Parameter = Union[sympy.Symbol, Number]

T = TypeVar("T", bound="Operation")


class Operation(Protocol):
    """Represents arbitrary operation applicable to a circuit or wavefunction."""

    @property
    @abstractmethod
    def params(self) -> Tuple[Parameter, ...]:
        """Parameters of this operation."""

    @abstractmethod
    def bind(self: T, symbols_map: Dict[sympy.Symbol, Parameter]) -> T:
        """Create new operation by replacing free symbols in operation params.

        The operation returned by this method should be of the same type
        as self, e.g. binding parameters to GateOperation should produce
        GateOperation.
        """

    @abstractmethod
    def replace_params(self: T, new_params: Tuple[Parameter, ...]) -> T:
        """Create new operation by replacing params.

        The difference between bind and replace params is that it bind performs
        parameter substitution - in particular, parameters without free symbols
        are unaffected by bind, whereas replace_params replaces *all* params.
        """

    @abstractmethod
    def apply(self: T, wavefunction: Sequence[Parameter]) -> Sequence[Parameter]:
        """Apply this operation to given wavefunction.

        Operations that cannot be applied deterministically should raise
        NotImplementedError.
        """

    @property
    @abstractmethod
    def free_symbols(self) -> Iterable[sympy.Symbol]:
        """Free symbols parameterizing this operation.

        Note that number of free_symbols is unrelated to number of params.
        Some params can be expressions with multiple free symbols, while other params
        might not comprise free symbols at all.
        """


@singledispatch
def sub_symbols(parameter, symbols_map: Dict[sympy.Symbol, Parameter]) -> Parameter:
    raise NotImplementedError()


@sub_symbols.register
def _sub_symbols_in_number(
    parameter: Number, symbols_map: Dict[sympy.Symbol, Parameter]
) -> Number:
    return parameter


@sub_symbols.register
def _sub_symbols_in_expression(
    parameter: sympy.Expr, symbols_map: Dict[sympy.Symbol, Parameter]
) -> sympy.Expr:
    return parameter.subs(symbols_map)


@sub_symbols.register
def _sub_symbols_in_symbol(
    parameter: sympy.Symbol, symbols_map: Dict[sympy.Symbol, Parameter]
) -> Parameter:
    return symbols_map.get(parameter, parameter)


def get_free_symbols(parameters: Tuple[Parameter, ...]) -> Iterable[sympy.Symbol]:
    symbols = set(
        symbol
        for param in parameters
        if isinstance(param, sympy.Expr)
        for symbol in param.free_symbols
    )
    return sorted(symbols, key=str)
