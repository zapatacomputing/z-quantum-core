from . import _gates as g
from . import _matrices as m


X = g.define_nonparametric_gate("X", matrix=m.x_matrix())
Y = g.define_nonparametric_gate("Y", matrix=m.y_matrix())
Z = g.define_nonparametric_gate("Z", matrix=m.z_matrix())

RX = g.define_one_param_gate("RX", matrix_factory=m.rx_matrix, n_qubits=1)
RY = g.define_one_param_gate("RY", matrix_factory=m.ry_matrix, n_qubits=1)
RZ = g.define_one_param_gate("RZ", matrix_factory=m.rz_matrix, n_qubits=1)
