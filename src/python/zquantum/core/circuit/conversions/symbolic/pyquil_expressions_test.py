"""Test cases for decomposing PyQuil expression into our native expressions."""
import pytest
from .pyquil_expressions import expression_from_pyquil


@pytest.mark.parametrize("number", [3, 4.0, 1j, 3.0 - 2j])
def test_native_numbers_are_preserved_when_decomposing_pyquil_expression(number):
    assert expression_from_pyquil(number) == number
