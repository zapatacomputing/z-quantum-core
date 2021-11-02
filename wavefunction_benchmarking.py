import sympy
from openfermion.ops.operators.ising_operator import IsingOperator
from zquantum.core.estimation import calculate_exact_expectation_values
from zquantum.core.interfaces.estimation import EstimationTask
from zquantum.core.openfermion import change_operator_type
from zquantum.core.symbolic_simulator import SymbolicSimulator
from zquantum.qaoa.ansatzes.farhi_ansatz import QAOAFarhiAnsatz
from zquantum.qaoa.problems.maxcut import MaxCut
from zquantum.core import circuits
from zquantum.core.graph import generate_random_graph_erdos_renyi
from qequlacs import QulacsSimulator

sim = SymbolicSimulator()
sim2 = QulacsSimulator()


def qaoa_circuit():
    H = MaxCut().get_hamiltonian(generate_random_graph_erdos_renyi(6, 0.8))
    H = change_operator_type(H, IsingOperator)
    ansatz = QAOAFarhiAnsatz(1, cost_hamiltonian=H)

    return ansatz.parametrized_circuit, H


sym_circuit, ham = qaoa_circuit()
print("prepare circuit...")

# wf = sim.get_wavefunction(sym_circuit)
# # wf = sympy.simplify(wf._amplitude_vector)
# breakpoint()
# fun = sympy.lambdify(tuple(wf.free_symbols), wf._amplitude_vector, "numpy")
# est_task = EstimationTask(ham, sym_circuit, 10)
# breakpoint()
# exp_vals = calculate_exact_expectation_values(sim, [est_task])[0].values[0]
# breakpoint()
