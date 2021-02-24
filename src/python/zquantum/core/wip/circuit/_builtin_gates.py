from . import _gates as g
from . import _matrices as m


X = g.Gate("X", m.x_matrix())
Y = g.Gate("Y", m.y_matrix())
Z = g.Gate("Z", m.z_matrix())


RX = g.make_one_param_gate_factory("RX", matrix_factory=m.rx_matrix)
RY = g.make_one_param_gate_factory("RY", matrix_factory=m.ry_matrix)
RZ = g.make_one_param_gate_factory("RZ", matrix_factory=m.rz_matrix)
