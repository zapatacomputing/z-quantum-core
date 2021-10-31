import jax
import jax.numpy as jnp
import numpy as np
import sympy
from jax import random
from openfermion.ops.operators.ising_operator import IsingOperator
from sympy import Symbol, adjoint
from sympy2jax import sympy2jax
from zquantum.core.circuits import RX, RY, RZ, Circuit
from zquantum.core.estimation import calculate_exact_expectation_values
from zquantum.core.interfaces.estimation import EstimationTask, EstimationTasksFactory
from zquantum.core.interfaces.functions import function_with_gradient
from zquantum.core.symbolic_simulator import SymbolicSimulator
from zquantum.core.wavefunction import Wavefunction
from zquantum.optimizers.simple_gradient_descent import SimpleGradientDescent


def exp_of_rx_gate_symbolic_openfermion():
    test_circuit = Circuit()
    gate = RX(2 * Symbol("theta"))(0)

    test_circuit += gate

    est_task = EstimationTask(IsingOperator("Z0"), test_circuit, 10000)

    return calculate_exact_expectation_values(sim, [est_task])[0].values[0]


sim = SymbolicSimulator()

symbolic_exp = exp_of_rx_gate_symbolic_openfermion()


f, params = sympy2jax(symbolic_exp, list(symbolic_exp.free_symbols))


def cost_function(curr_params):
    return f(curr_params, params)[0]


fun = function_with_gradient(cost_function, jax.grad(cost_function))

grad_desc = SimpleGradientDescent(0.1, 100)

key = random.PRNGKey(0)
X = random.normal(key, (1, 1))

res = grad_desc.minimize(fun, X, keep_history=False)

print(res)
