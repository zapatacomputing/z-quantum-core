from . import _gates as g
from . import _matrices as m


X = g.Gate("X", m.x_matrix())
Y = g.Gate("Y", m.y_matrix())
Z = g.Gate("Z", m.z_matrix())
H = g.Gate("H", m.h_matrix())
I = g.Gate("I", m.i_matrix())
T = g.Gate("T", m.t_matrix())

RX = g.make_one_param_gate_factory("RX", matrix_factory=m.rx_matrix)
RY = g.make_one_param_gate_factory("RY", matrix_factory=m.ry_matrix)
RZ = g.make_one_param_gate_factory("RZ", matrix_factory=m.rz_matrix)
PHASE = g.make_one_param_gate_factory("PHASE", matrix_factory=m.phase_matrix)
