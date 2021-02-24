from . import _gates as g
from . import _matrices as m


# --- non-parametric, single qubit gates ---

X = g.Gate("X", m.x_matrix())
Y = g.Gate("Y", m.y_matrix())
Z = g.Gate("Z", m.z_matrix())
H = g.Gate("H", m.h_matrix())
I = g.Gate("I", m.i_matrix())
T = g.Gate("T", m.t_matrix())


# --- parametric, single qubit gates ---

RX = g.make_parametric_gate_factory("RX", matrix_factory=m.rx_matrix)
RY = g.make_parametric_gate_factory("RY", matrix_factory=m.ry_matrix)
RZ = g.make_parametric_gate_factory("RZ", matrix_factory=m.rz_matrix)
PHASE = g.make_parametric_gate_factory("PHASE", matrix_factory=m.phase_matrix)


# --- non-parametric, two qubit gates ---

CNOT = g.Gate("CNOT", m.cnot_matrix())
CZ = g.Gate("CZ", m.cz_matrix())
SWAP = g.Gate("SWAP", m.swap_matrix())
ISWAP = g.Gate("ISWAP", m.iswap_matrix())


# --- parametric, two qubit gates ---

CPHASE = g.make_parametric_gate_factory("CPHASE", matrix_factory=m.cphase_matrix)
XX = g.make_parametric_gate_factory("XX", matrix_factory=m.xx_matrix)
YY = g.make_parametric_gate_factory("YY", matrix_factory=m.yy_matrix)
ZZ = g.make_parametric_gate_factory("ZZ", matrix_factory=m.zz_matrix)
XY = g.make_parametric_gate_factory("XY", matrix_factory=m.xy_matrix)
