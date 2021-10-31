import networkx as nx
import sympy
from jax import grad, jit, random
from numpy import argpartition
from openfermion.ops.operators.ising_operator import IsingOperator
from sympy2jax import sympy2jax
from zquantum.core.estimation import calculate_exact_expectation_values
from zquantum.core.interfaces.estimation import EstimationTask
from zquantum.core.interfaces.functions import function_with_gradient
from zquantum.core.openfermion import change_operator_type
from zquantum.core.symbolic_simulator import SymbolicSimulator
from zquantum.optimizers.simple_gradient_descent import SimpleGradientDescent
from zquantum.qaoa.ansatzes.farhi_ansatz import QAOAFarhiAnsatz
from zquantum.qaoa.problems.maxcut import MaxCut

sim = SymbolicSimulator()


def exp_qaoa():
    G = nx.Graph()
    # Need to be very small graph for performance issues
    G.add_nodes_from([0, 1, 2])
    G.add_edge(0, 1, weight=10)
    G.add_edge(0, 2, weight=10)
    G.add_edge(1, 2, weight=1)
    H = MaxCut().get_hamiltonian(G)
    H = change_operator_type(H, IsingOperator)
    ansatz = QAOAFarhiAnsatz(1, cost_hamiltonian=H)

    circuit = ansatz.parametrized_circuit

    est_task = EstimationTask(H, circuit, 10)

    return circuit, calculate_exact_expectation_values(sim, [est_task])[0].values[0]


# .re is needed here to avoid complex-typed values in the cost
# and make it compatible with JAX
circuit, symbolic_exp = exp_qaoa()
print("Ran circuit...")

symbolic_exp = sympy.re(sympy.simplify(symbolic_exp))
print("Simplified and Real-ized symbolic expectation expression...")

f, params = sympy2jax(symbolic_exp, list(symbolic_exp.free_symbols))

lamb_f = lambda x, y: f(x, y)[0]
compiled_f = jit(lamb_f)
compiled_f_grad = grad(compiled_f)

compiled_f_with_params = lambda x: compiled_f(x, params)
compiled_f_grad_with_params = lambda x: compiled_f_grad(x, params)
key = random.PRNGKey(0)
X = random.normal(key, (1, len(symbolic_exp.free_symbols)))
print("Transformed to JAX function...")

fun = function_with_gradient(compiled_f_with_params, compiled_f_grad_with_params)

grad_desc = SimpleGradientDescent(
    0.1,
    100,
)

print("Prepared for optimization...")

res = grad_desc.minimize(fun, X, keep_history=False)
print("Finished optimization...")

wf = (
    sim.get_wavefunction(
        circuit.bind(dict(zip(symbolic_exp.free_symbols, res.opt_params[0])))
    )
    .probabilities()
    .flatten()
)

print(f"Top 2 measurements (not 0 padded): {list(map(bin, argpartition(wf, -2)[-2:]))}")
breakpoint()
