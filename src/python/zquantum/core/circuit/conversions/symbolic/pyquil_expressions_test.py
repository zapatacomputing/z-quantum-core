"""Test cases for decomposing PyQuil expression into our native expressions."""
from pyquil import quil, quilatom
import pytest
from .expressions import Symbol
from .pyquil_expressions import expression_from_pyquil


@pytest.mark.parametrize("number", [3, 4.0, 1j, 3.0 - 2j])
def test_native_numbers_are_preserved_when_decomposing_pyquil_expression(number):
    assert expression_from_pyquil(number) == number


@pytest.mark.parametrize(
    "pyquil_parameter, expected_symbol",
    [
        (quil.Parameter("theta"), Symbol("theta")),
        (quil.Parameter("x"),  Symbol("x")),
        (quil.Parameter("x_1"), Symbol("x_1"))
    ]
)
def test_quil_parameters_are_converted_to_instance_of_symbol_with_correct_name(
    pyquil_parameter, expected_symbol
):
    assert expression_from_pyquil(pyquil_parameter) == expected_symbol
