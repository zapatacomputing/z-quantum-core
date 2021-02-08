import operator
from functools import reduce

from .expressions import ExpressionDialect, FunctionCall
import qiskit

from .helpers import reduction


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
