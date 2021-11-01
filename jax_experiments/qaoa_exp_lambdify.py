import networkx as nx
import sympy
from jax import grad, random
from numpy import argpartition
from openfermion.ops.operators.ising_operator import IsingOperator
from zquantum.core.estimation import calculate_exact_expectation_values
from zquantum.core.interfaces.estimation import EstimationTask
from zquantum.core.interfaces.functions import function_with_gradient
from zquantum.core.openfermion import change_operator_type
from zquantum.core.symbolic_simulator import SymbolicSimulator
from zquantum.optimizers.simple_gradient_descent import SimpleGradientDescent
from zquantum.qaoa.ansatzes.farhi_ansatz import QAOAFarhiAnsatz
from zquantum.qaoa.problems.maxcut import MaxCut
import jax.numpy as jnp
from jax.scipy import special as jsp

sim = SymbolicSimulator()

"""
WARNING: THIS CODE IS STILL NON-FUNCTIONING
"""


_jnp_func_lookup = {
    "sympy.Mul": jnp.multiply,
    "sympy.Add": jnp.add,
    "sympy.div": jnp.divide,
    "sympy.Abs": jnp.abs,
    "sympy.sign": jnp.sign,
    # Note: May raise error for ints.
    "sympy.ceiling": jnp.ceil,
    "sympy.floor": jnp.floor,
    "sympy.log": jnp.log,
    "sympy.exp": jnp.exp,
    "sympy.sqrt": jnp.sqrt,
    "cos": jnp.cos,
    "sympy.acos": jnp.arccos,
    "sympy.sin": jnp.sin,
    "sympy.asin": jnp.arcsin,
    "sympy.tan": jnp.tan,
    "sympy.atan": jnp.arctan,
    "sympy.atan2": jnp.arctan2,
    # Note: Also may give NaN for complex results.
    "sympy.cosh": jnp.cosh,
    "sympy.acosh": jnp.arccosh,
    "sympy.sinh": jnp.sinh,
    "sympy.asinh": jnp.arcsinh,
    "sympy.tanh": jnp.tanh,
    "sympy.atanh": jnp.arctanh,
    "sympy.Pow": jnp.power,
    "sympy.re": jnp.real,
    "sympy.im": jnp.imag,
    # Note: May raise error for ints and complexes
    "sympy.erf": jsp.erf,
    "sympy.erfc": jsp.erfc,
    "sympy.LessThan": jnp.less,
    "sympy.GreaterThan": jnp.greater,
    "sympy.And": jnp.logical_and,
    "sympy.Or": jnp.logical_or,
    "sympy.Not": jnp.logical_not,
    "sympy.Max": jnp.fmax,
    "sympy.Min": jnp.fmin,
    "sympy.Mod": jnp.fmod,
    "sympy.conjugate": jnp.conjugate,
}


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
import inspect

fun = sympy.lambdify(
    list(symbolic_exp.free_symbols), symbolic_exp, modules=_jnp_func_lookup
)
lamb_f = lambda x: fun(*x[0])
key = random.PRNGKey(0)
X = random.normal(key, (1, len(symbolic_exp.free_symbols)))
lines = inspect.getsource(fun)
breakpoint()
print("Transformed to JAX function...")

fun = function_with_gradient(lamb_f, grad(lamb_f))

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
