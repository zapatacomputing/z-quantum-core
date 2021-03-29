"""Translations between Qiskit parameter expressions and intermediate expression trees.

Attributes:
    QISKIT_DIALECT: Mapping from the intermediate expression tree into atoms
        used in Qiskit symbolic expressions. Allows translating an expression
        into the Qiskit dialect. Can be used with
        `zquantum.core.circuit.symbolic.translations.translate_expression`.
"""
import operator
from functools import reduce, singledispatch
from numbers import Number

import qiskit

from .expressions import ExpressionDialect, reduction
from .sympy_expressions import expression_from_sympy


@singledispatch
def expression_from_qiskit(expression):
    """Parse Qiskit expression into intermediate expression tree."""
    raise NotImplementedError(
        f"Expression {expression} of type {type(expression)} is currently not supported"
    )


@expression_from_qiskit.register
def _number_identity(number: Number):
    return number


@expression_from_qiskit.register
def _expr_from_qiskit_param_expr(
    qiskit_expr: qiskit.circuit.parameterexpression.ParameterExpression,
):
    # At the moment of writing this the qiskit version that we use (0.23.2) as well
    # as the newest version 0.23.5) does not provide a better way to access symbolic
    # expression wrapped by ParameterExpression.
    sympy_expr = qiskit_expr._symbol_expr
    return expression_from_sympy(sympy_expr)


def integer_pow(base, exponent: int):
    """Exponentiation to the power of an integer exponent."""
    if not isinstance(exponent, int):
        raise ValueError(
            f"Cannot convert expression {base} ** {exponent} to Qiskit. "
            "Only powers with integral exponent are convertible."
        )
    if exponent < 0:
        if base != 0:
            base = 1 / base
            exponent = -exponent
        else:
            raise ValueError(
                f"Invalid power: cannot raise 0 to exponent {exponent} < 0."
            )
    return reduce(operator.mul, exponent * [base], 1)


QISKIT_DIALECT = ExpressionDialect(
    symbol_factory=lambda symbol: qiskit.circuit.Parameter(symbol.name),
    number_factory=lambda number: number,
    known_functions={
        "add": reduction(operator.add),
        "mul": reduction(operator.mul),
        "div": operator.truediv,
        "sub": operator.sub,
        "pow": integer_pow,
    },
)
