from zquantum.core.circuits._gates import GateOperation
from zquantum.core.circuits._circuit import Circuit


def sk_decomposition(U: GateOperation, n: int) -> Circuit:
    if n == 0:
        return simple_approx(U)
    U_new = sk_decomposition(U, n - 1)
    V, W = GC_decompose(U * U_new.dagger())
    V_new = sk_decomposition(V, n - 1)
    W_new = sk_decomposition(W, n - 1)
    return V * W * V_new.dagger() * W_new.dagger() * U_new


def simple_approx(U: GateOperation):
    pass


def GC_decompose(U):
    pass
