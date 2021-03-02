from typing import Callable

from . import _gates as g
from . import _matrices as m


# --- non-parametric, single qubit gates ---

X = g.MatrixFactoryGate("X", m.x_matrix, (), 1, is_hermitian=True)
Y = g.MatrixFactoryGate("Y", m.y_matrix, (), 1, is_hermitian=True)
Z = g.MatrixFactoryGate("Z", m.z_matrix, (), 1, is_hermitian=True)
H = g.MatrixFactoryGate("H", m.h_matrix, (), 1, is_hermitian=True)
I = g.MatrixFactoryGate("I", m.i_matrix, (), 1, is_hermitian=True)
T = g.MatrixFactoryGate("T", m.t_matrix, (), 1)


# gate1.namespace == gate2.namespace and gate1.name == gate2.name


# ("X", "zquantum.core.wip.circuit._builtin_gates")
# ("X", "zqe.basf.gates")
# ("X")


# # problem 2

# X = def_gate("X", [[0, 1], [1, 0]])
# Y = def_gate("X", [[0, 1], [1, 0]])
# Z = def_gate("X", [[-2, -2], [-2, -2]])

# X == Y ?
# X == Z ?


#     return f"{gate.namespace}.{gate.name}"


# # 1. zquantum built-in gate <-> pyquil built-in gate
# {
#     "name": "X",
#     "namespace": "zquantum",
#     "namespace": "",
#     "namespace": None,
#     "namespace": __module__,
# }
# <->
# pyquil.X

# # 2. zquantum built-in gate <-> pyquil custom gate
# {
#     "name": "XX",
#     "namespace": "zquantum",
#     # "namespace": __module__,
#     "namespace": "zquantum.gates",
# }
# <->
# # pyquil.DefGate("zquantum.XX")
# # pyquil.DefGate("zquantum.core.circuit._builtin_gates.XX")
# pyquil.DefGate("zquantum.gates.XX")

# # 3. zquantum custom gate <-> pyquil custom gate
# {
#     "name": "G",
#     "namespace": "coca-cola.gates",
# }
# pyquil.DefGate("coca-cola.G")


# --- parametric, single qubit gates ---

def make_parametric_gate_factory(name, matrix_factory, num_qubits) -> Callable[..., g.MatrixFactoryGate]:
    def _factory(*gate_parameters):
        # TODO: check if len(gate_parameters) == len(arguments of matrix_factory)
        return g.MatrixFactoryGate(name, matrix_factory, gate_parameters, num_qubits)
    return _factory


RX = make_parametric_gate_factory("RX", m.rx_matrix, 1)
RY = make_parametric_gate_factory("RY", m.ry_matrix, 1)
RZ = make_parametric_gate_factory("RZ", m.rz_matrix, 1)
PHASE = make_parametric_gate_factory("PHASE", m.phase_matrix, 1)


# --- non-parametric, two qubit gates ---

CNOT = g.MatrixFactoryGate("CNOT", m.cnot_matrix, (), 2, is_hermitian=True)
CZ = g.MatrixFactoryGate("CZ", m.cz_matrix, (), 2, is_hermitian=True)
SWAP = g.MatrixFactoryGate("SWAP", m.swap_matrix, (), 2, is_hermitian=True)
ISWAP = g.MatrixFactoryGate("ISWAP", m.iswap_matrix, (), 2)


# --- parametric, two qubit gates ---

CPHASE = make_parametric_gate_factory("CPHASE", m.cphase_matrix, 2)
XX = make_parametric_gate_factory("XX", m.xx_matrix, 2)
YY = make_parametric_gate_factory("YY", m.yy_matrix, 2)
ZZ = make_parametric_gate_factory("ZZ", m.zz_matrix, 2)
XY = make_parametric_gate_factory("XY", m.xy_matrix, 2)
