"""Utilities for converting symbolic expressions between different dialects."""
from functools import singledispatch
from numbers import Number
from typing import NamedTuple, Any, Iterable, Union, Dict, Callable

import sympy


class Symbol(NamedTuple):
    """Abstract symbol."""

    name: str


class FunctionCall(NamedTuple):
    """Represents abstract function call.     """

    name: str
    args: Iterable["Expression"]


Expression = Union[Symbol, FunctionCall, Number]


def is_multiplication_by_reciprocal(sympy_mul: sympy.Mul) -> bool:
    """Check if given sympy multiplication is of the form x * (1 / y)."""
    args = sympy_mul.args
    return len(args) == 2 and isinstance(args[1], sympy.Pow) and args[1].args[1] == -1


def is_addition_of_negation(sympy_add: sympy.Add) -> bool:
    """Check if given sympy addition os of the form x + (-y)."""
    args = sympy_add.args
    return len(args) == 2 and isinstance(args[1], sympy.Mul) and args[1].args[0] == -1


@singledispatch
def expression_tree_from_sympy(expression):
    raise NotImplementedError(
        f"Expression {expression} of type {type(expression)} is currentlyl not supported"
    )


@expression_tree_from_sympy.register
def symbol_from_sympy(symbol: sympy.Symbol):
    return Symbol(str(symbol))


@expression_tree_from_sympy.register
def native_number_from_sympy_number(number: sympy.Number):
    return number.n()


@expression_tree_from_sympy.register
def native_imaginary_unit_from_sympy_imaginary_unit(_unit: sympy.numbers.ImaginaryUnit):
    return 1j


@expression_tree_from_sympy.register
def addition_from_sympy_add(add: sympy.Add):
    if is_addition_of_negation(add):
        return FunctionCall(
            "sub",
            (
                expression_tree_from_sympy(add.args[0]),
                expression_tree_from_sympy(add.args[1].args[1]),
            ),
        )
    return FunctionCall(
        "add", tuple(expression_tree_from_sympy(arg) for arg in add.args)
    )


@expression_tree_from_sympy.register
def multiplication_from_sympy_mul(mul: sympy.Mul):
    if is_multiplication_by_reciprocal(mul):
        return FunctionCall(
            "div",
            (
                expression_tree_from_sympy(mul.args[0]),
                expression_tree_from_sympy(mul.args[1].args[0]),
            ),
        )
    else:
        return FunctionCall(
            "mul", tuple(expression_tree_from_sympy(arg) for arg in mul.args)
        )


@expression_tree_from_sympy.register
def power_from_sympy_pow(power: sympy.Pow):
    if power.args[1] == -1:
        return FunctionCall("div", (1, expression_tree_from_sympy(power.args[0])))
    return FunctionCall(
        "pow", tuple(expression_tree_from_sympy(arg) for arg in power.args)
    )
