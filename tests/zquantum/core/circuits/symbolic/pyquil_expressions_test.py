"""Test cases for decomposing PyQuil expression into our native expressions."""
import numpy as np
import pytest
from pyquil import quil, quilatom
from zquantum.core.circuits.symbolic.expressions import FunctionCall, Symbol
from zquantum.core.circuits.symbolic.pyquil_expressions import expression_from_pyquil


@pytest.mark.parametrize("number", [3, 4.0, 1j, 3.0 - 2j])
def test_native_numbers_are_preserved(number):
    assert expression_from_pyquil(number) == number


@pytest.mark.parametrize(
    "pyquil_parameter, expected_symbol",
    [
        (quil.Parameter("theta"), Symbol("theta")),
        (quil.Parameter("x"), Symbol("x")),
        (quil.Parameter("x_1"), Symbol("x_1")),
    ],
)
def test_quil_parameters_are_converted_to_instance_of_symbol_with_correct_name(
    pyquil_parameter, expected_symbol
):
    assert expression_from_pyquil(pyquil_parameter) == expected_symbol


@pytest.mark.parametrize(
    "pyquil_function_call, expected_function_call",
    [
        (quilatom.quil_cos(2), FunctionCall("cos", (2,))),
        (
            quilatom.quil_sin(quil.Parameter("theta")),
            FunctionCall("sin", (Symbol("theta"),)),
        ),
        (quilatom.quil_exp(quil.Parameter("x")), FunctionCall("exp", (Symbol("x"),))),
        (quilatom.quil_sqrt(np.pi), FunctionCall("sqrt", (np.pi,))),
    ],
)
def test_pyquil_function_calls_are_converted_to_equivalent_function_call(
    pyquil_function_call, expected_function_call
):
    assert expression_from_pyquil(pyquil_function_call) == expected_function_call


@pytest.mark.parametrize(
    "pyquil_expression, expected_function_call",
    [
        (
            quil.Parameter("x") + quil.Parameter("y"),
            FunctionCall("add", (Symbol("x"), Symbol("y"))),
        ),
        (
            quilatom.quil_cos(quil.Parameter("theta")) * 2,
            FunctionCall("mul", (FunctionCall("cos", (Symbol("theta"),)), 2)),
        ),
        (
            quilatom.quil_sqrt(quil.Parameter("phi")) / quil.Parameter("psi"),
            FunctionCall(
                "div", (FunctionCall("sqrt", (Symbol("phi"),)), Symbol("psi"))
            ),
        ),
        (
            quil.Parameter("a") - quil.Parameter("b"),
            FunctionCall("sub", (Symbol("a"), Symbol("b"))),
        ),
        (2 ** quil.Parameter("N"), FunctionCall("pow", (2, Symbol("N")))),
    ],
)
def test_pyquil_binary_expressions_are_converted_to_appropriate_function_call(
    pyquil_expression, expected_function_call
):
    assert expression_from_pyquil(pyquil_expression) == expected_function_call
