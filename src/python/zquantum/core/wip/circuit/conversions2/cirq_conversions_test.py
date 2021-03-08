"""Test cases from Orquestra <-> cirq conversions."""
import cirq
import pytest

from .cirq_conversions import convert_from_cirq, convert_to_cirq
from .. import _builtin_gates as bg


@pytest.mark.parametrize(
    "orquestra_operation, cirq_operation",
    [
        (bg.X(1), cirq.X(cirq.LineQubit(1))),
        (bg.Y(2), cirq.Y(cirq.LineQubit(2))),
        (bg.Z(0), cirq.Z(cirq.LineQubit(0))),
        (bg.I(5), cirq.I(cirq.LineQubit(5))),
        (bg.H(0), cirq.H(cirq.LineQubit(0))),
        (bg.T(1), cirq.T(cirq.LineQubit(1))),
        (bg.CNOT(1, 2), cirq.CNOT(cirq.LineQubit(1), cirq.LineQubit(2))),
        (bg.CZ(1, 0), cirq.CZ(cirq.LineQubit(1), cirq.LineQubit(0))),
        (bg.SWAP(2, 0), cirq.SWAP(cirq.LineQubit(2), cirq.LineQubit(0)))
    ]
)
class TestBuiltinGateOperationConversion:

    def test_orquestra_builtin_gate_operation_is_converted_to_its_cirq_counterpart(
        self,
        orquestra_operation,
        cirq_operation
    ):
        assert convert_to_cirq(orquestra_operation) == cirq_operation

    def test_cirq_gate_operation_is_converted_to_its_orquestra_counterpart(
        self,
        cirq_operation,
        orquestra_operation
    ):
        assert convert_from_cirq(cirq_operation) == orquestra_operation
