"""Test cases for _builtin_gates_module."""
import pytest

from . import _builtin_gates


class TestBuiltinGatesProperties:
    @pytest.mark.parametrize(
        "gate",
        [
            _builtin_gates.X,
            _builtin_gates.Y,
            _builtin_gates.Z,
            _builtin_gates.CZ,
            _builtin_gates.CNOT,
            _builtin_gates.I,
            _builtin_gates.T,
            _builtin_gates.H,
            _builtin_gates.S,
            _builtin_gates.SWAP,
            _builtin_gates.ISWAP,
            _builtin_gates.RX(0.5),
            _builtin_gates.RY(1),
            _builtin_gates.RZ(0.5),
            _builtin_gates.XX(0.1),
            _builtin_gates.YY(0.2),
            _builtin_gates.ZZ(0.3),
            _builtin_gates.PHASE(1),
            _builtin_gates.CPHASE(0.1),
        ],
    )
    def test_gates_matrix_equals_its_adjoint_iff_gate_is_hermitian(self, gate):
        assert (gate.matrix == gate.matrix.adjoint()) == gate.is_hermitian
