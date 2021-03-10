from typing import Callable, Union, Optional

from . import _gates as g
from . import _matrices as m


GatePrototype = Callable[..., g.MatrixFactoryGate]
GateRef = Union[g.Gate, GatePrototype]


def make_parametric_gate_prototype(name, matrix_factory, num_qubits) -> GatePrototype:
    def _factory(*gate_parameters):
        # TODO: check if len(gate_parameters) == len(arguments of matrix_factory)
        return g.MatrixFactoryGate(name, matrix_factory, gate_parameters, num_qubits)
    return _factory


def builtin_gate_by_name(name) -> Optional[GateRef]:
    return globals().get(name)


# --- non-parametric, single qubit gates ---

X = g.MatrixFactoryGate("X", m.x_matrix, (), 1, is_hermitian=True)
Y = g.MatrixFactoryGate("Y", m.y_matrix, (), 1, is_hermitian=True)
Z = g.MatrixFactoryGate("Z", m.z_matrix, (), 1, is_hermitian=True)
H = g.MatrixFactoryGate("H", m.h_matrix, (), 1, is_hermitian=True)
I = g.MatrixFactoryGate("I", m.i_matrix, (), 1, is_hermitian=True)
T = g.MatrixFactoryGate("T", m.t_matrix, (), 1)


# --- parametric, single qubit gates ---


RX = make_parametric_gate_prototype("RX", m.rx_matrix, 1)
RY = make_parametric_gate_prototype("RY", m.ry_matrix, 1)
RZ = make_parametric_gate_prototype("RZ", m.rz_matrix, 1)
PHASE = make_parametric_gate_prototype("PHASE", m.phase_matrix, 1)


# --- non-parametric, two qubit gates ---

CNOT = g.MatrixFactoryGate("CNOT", m.cnot_matrix, (), 2, is_hermitian=True)
CZ = g.MatrixFactoryGate("CZ", m.cz_matrix, (), 2, is_hermitian=True)
SWAP = g.MatrixFactoryGate("SWAP", m.swap_matrix, (), 2, is_hermitian=True)
ISWAP = g.MatrixFactoryGate("ISWAP", m.iswap_matrix, (), 2)


# --- parametric, two qubit gates ---

CPHASE = make_parametric_gate_prototype("CPHASE", m.cphase_matrix, 2)
XX = make_parametric_gate_prototype("XX", m.xx_matrix, 2)
YY = make_parametric_gate_prototype("YY", m.yy_matrix, 2)
ZZ = make_parametric_gate_prototype("ZZ", m.zz_matrix, 2)
XY = make_parametric_gate_prototype("XY", m.xy_matrix, 2)
