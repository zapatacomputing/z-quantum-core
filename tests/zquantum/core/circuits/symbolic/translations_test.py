"""Test cases for symbolic_expressions module."""
import pytest
import sympy
from zquantum.core.circuits.symbolic.sympy_expressions import (
    SYMPY_DIALECT,
    expression_from_sympy,
)
from zquantum.core.circuits.symbolic.translations import translate_expression


@pytest.mark.parametrize(
    "sympy_expression",
    [
        sympy.Symbol("theta"),
        sympy.Symbol("theta") * sympy.Symbol("gamma"),
        sympy.cos(sympy.Symbol("theta")),
        sympy.cos(2 * sympy.Symbol("theta")),
        sympy.exp(sympy.Symbol("x") - sympy.Symbol("y")),
        sympy.cos("phi") + sympy.I * sympy.sin(sympy.Symbol("phi")),
        sympy.Symbol("x") + sympy.Symbol("y") * (2 + 3j),
        sympy.cos(sympy.sin(sympy.Symbol("tau"))),
        sympy.Symbol("x") / sympy.Symbol("y"),
        sympy.tan(sympy.Symbol("theta")),
        2 ** sympy.Symbol("x"),
        sympy.Symbol("y") ** sympy.Symbol("x"),
        sympy.Symbol("x") ** 2,
        sympy.sqrt(sympy.Symbol("x") - sympy.Symbol("y")),
        -5 * sympy.Symbol("x") * sympy.Symbol("y"),
        sympy.Symbol("x") + sympy.Symbol("y") + 9,
    ],
)
def test_translating_tree_from_sympy_to_quil_gives_expected_result(sympy_expression):
    expression = expression_from_sympy(sympy_expression)
    assert translate_expression(expression, SYMPY_DIALECT) - sympy_expression == 0
