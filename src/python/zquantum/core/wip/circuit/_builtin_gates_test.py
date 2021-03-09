"""Test cases for _builtin_gates_module."""
import pytest

from . import _builtin_gates as bg


class TestBuiltinGatesProperties:
    @pytest.mark.parametrize(
        "gate",
        [
            bg.X,
            bg.Y,
            bg.Z,
            bg.CZ,
            bg.CNOT,
            bg.I,
            bg.T,
            bg.H,
            bg.SWAP,
            bg.ISWAP,
            bg.RX(0.5),
            bg.RY(1),
            bg.RZ(0.5),
            bg.XX(0.1),
            bg.YY(0.2),
            bg.ZZ(0.3),
            bg.PHASE(1),
            bg.CPHASE(0.1),
        ],
    )
    def test_gates_matrix_equals_its_adjoint_iff_gate_is_hermitian(self, gate):
        assert (gate.matrix == gate.matrix.adjoint()) == gate.is_hermitian
