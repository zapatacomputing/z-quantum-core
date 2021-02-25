from typing import Callable

from . import _gates as g
from . import _matrices as m


# --- non-parametric, single qubit gates ---

X = g.MatrixFactoryGate("X", m.x_matrix, (), 1)
Y = g.MatrixFactoryGate("Y", m.y_matrix, (), 1)
Z = g.MatrixFactoryGate("Z", m.z_matrix, (), 1)
H = g.MatrixFactoryGate("H", m.h_matrix, (), 1)
I = g.MatrixFactoryGate("I", m.i_matrix, (), 1)
T = g.MatrixFactoryGate("T", m.t_matrix, (), 1)


# --- parametric, single qubit gates ---

def make_parametric_gate_factory(name, matrix_factory, num_qubits) -> Callable[..., g.MatrixFactoryGate]:
    def _factory(*gate_parameters):
        return g.MatrixFactoryGate(name, matrix_factory, gate_parameters, num_qubits)
    return _factory


RX = make_parametric_gate_factory("RX", m.rx_matrix, 1)
RY = make_parametric_gate_factory("RY", m.ry_matrix, 1)
RZ = make_parametric_gate_factory("RZ", m.rz_matrix, 1)
PHASE = make_parametric_gate_factory("PHASE", m.phase_matrix, 1)


# --- non-parametric, two qubit gates ---

CNOT = g.MatrixFactoryGate("CNOT", m.cnot_matrix, (), 2)
CZ = g.MatrixFactoryGate("CZ", m.cz_matrix, (), 2)
SWAP = g.MatrixFactoryGate("SWAP", m.swap_matrix, (), 2)
ISWAP = g.MatrixFactoryGate("ISWAP", m.iswap_matrix, (), 2)


# --- parametric, two qubit gates ---

CPHASE = make_parametric_gate_factory("CPHASE", m.cphase_matrix, 2)
XX = make_parametric_gate_factory("XX", m.xx_matrix, 2)
YY = make_parametric_gate_factory("YY", m.yy_matrix, 2)
ZZ = make_parametric_gate_factory("ZZ", m.zz_matrix, 2)
XY = make_parametric_gate_factory("XY", m.xy_matrix, 2)
