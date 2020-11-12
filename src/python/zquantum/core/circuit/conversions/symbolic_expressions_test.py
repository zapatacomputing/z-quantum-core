"""Test cases for symbolic_expressions module."""
from pyquil import quil, quilatom
import sympy
import pytest
from .sympy_expressions import expression_from_sympy
from .symbolic_expressions import translate_expression
from .pyquil_expressions import QUIL_DIALECT


@pytest.mark.parametrize(
    "sympy_expression, quil_expression",
    [
        (sympy.Symbol("theta"), quil.Parameter("theta")),
        (
            sympy.Mul(sympy.Symbol("theta"), sympy.Symbol("gamma"), evaluate=False),
            quil.Parameter("theta") * quil.Parameter("gamma"),
        ),
        (sympy.cos(sympy.Symbol("theta")), quilatom.quil_cos(quil.Parameter("theta"))),
        (sympy.cos(2 * sympy.Symbol("theta")), quilatom.quil_cos(2 * quil.Parameter("theta"))),
        (
                sympy.exp(sympy.Symbol("x") - sympy.Symbol("y")),
                quilatom.quil_exp(quil.Parameter("x") - quil.Parameter("y"))
        ),
        (
            sympy.Add(
                sympy.cos(sympy.Symbol("phi")),
                sympy.I * sympy.sin(sympy.Symbol("phi")),
                evaluate=False
            ),
            quilatom.quil_cos(quil.Parameter("phi")) + 1j * quilatom.quil_sin(quil.Parameter("phi"))
        ),
        (
            sympy.Add(
                sympy.Symbol("x"),
                sympy.Mul(sympy.Symbol("y"), (2 + 3j), evaluate=False),
                evaluate=False
            ),
            quil.Parameter("x") + quil.Parameter("y") * (2 + 3j)
        ),
        (
            sympy.cos(sympy.sin(sympy.Symbol("tau"))),
            quilatom.quil_cos(quilatom.quil_sin(quil.Parameter("tau")))
        ),
        (
            sympy.Symbol("x") / sympy.Symbol("y"),
            quil.Parameter("x") / quil.Parameter("y")
        ),
        (
            sympy.tan(sympy.Symbol("theta")),
            quilatom.quil_sin(quil.Parameter("theta")) / quilatom.quil_cos(quil.Parameter("theta"))
        ),
        (
            2 ** sympy.Symbol("x"),
            2 ** quil.Parameter("x")
        ),
        (
            sympy.Symbol("y") ** sympy.Symbol("x"),
            quil.Parameter("y") ** quil.Parameter("x")
        ),
        (
            sympy.Symbol("x") ** 2,
            quil.Parameter("x") ** 2
        ),
        (
            sympy.sqrt(sympy.Symbol("x") - sympy.Symbol("y")),
            quilatom.quil_sqrt(quil.Parameter("x") - quil.Parameter("y"))
        )
    ]
)
def test_translating_tree_from_sympy_to_quil_gives_expected_result(
    sympy_expression, quil_expression
):
    expression = expression_from_sympy(sympy_expression)
    assert translate_expression(expression, QUIL_DIALECT) == quil_expression
