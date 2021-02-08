import pytest
import qiskit

from .expressions import FunctionCall
from .qiskit_expressions import integer_pow


THETA = qiskit.circuit.Parameter("theta")


class TestIntegerPower:

    @pytest.mark.parametrize(
        "base, exponent, expected_result",
        [
            (2.5, 3, 2.5 ** 3),
            (THETA, 2, THETA * THETA)
        ]
    )
    def test_integer_power_with_positive_exponent_is_converted_to_repeated_multiplication(
        self, base, exponent, expected_result
    ):
        power = FunctionCall("pow", (base, exponent))
        assert integer_pow(power) == expected_result
