from . import _gates as g
from . import _matrices as m


X = g.define_nonparametric_gate("X", m.x_matrix())
Y = g.define_nonparametric_gate("Y", m.x_matrix())
Z = g.define_nonparametric_gate("Z", m.x_matrix())


def RX(angle):
    return g.CustomGate(
        name="RX", matrix_factory=m.rx_matrix, params=(angle,), num_qubits=1
    )


def RY(angle):
    return g.CustomGate(
        name="RY", matrix_factory=m.ry_matrix, params=(angle,), num_qubits=1
    )


def RZ(angle):
    return g.CustomGate(
        name="RZ", matrix_factory=m.rz_matrix, params=(angle,), num_qubits=1
    )
