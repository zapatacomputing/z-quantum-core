import operator
from functools import reduce, singledispatch
from numbers import Number

import qiskit

from .expressions import ExpressionDialect, FunctionCall, Symbol
from .helpers import reduction
from .sympy_expressions import expression_from_sympy


@singledispatch
def expression_from_qiskit(expression):
    '''Parse Qiskit expression into intermediate expression tree.'''
    raise NotImplementedError(
        f"Expression {expression} of type {type(expression)} is currently not supported"
    )


@expression_from_qiskit.register
def _number_identity(number: Number):
    return number


@expression_from_qiskit.register
def _expr_from_qiskit_param_expr(qiskit_expr: qiskit.circuit.parameterexpression.ParameterExpression):
    # At the moment of writing this (qiskit==0.16.1) there's no other way to introspect
    # a Qiskit parameter expression than by using a property from the private API.
    sympy_expr = qiskit_expr._symbol_expr
    return expression_from_sympy(sympy_expr)



def integer_pow(pow_call: FunctionCall):
    base, exponent = pow_call.args
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


# A mapping from the intermediate expression tree into atoms used in Qiskit symbolic expressions.
# Allows translating an expression into the Qiskit dialect.
QISKIT_DIALECT = ExpressionDialect(
    symbol_factory=lambda symbol: qiskit.circuit.Parameter(symbol.name),
    number_factory=lambda number: number,
    known_functions={
        "add": reduction(operator.add),
        "mul": reduction(operator.mul),
        "div": operator.truediv,
        "sub": operator.sub,
        "pow": operator.pow,
    }
)
