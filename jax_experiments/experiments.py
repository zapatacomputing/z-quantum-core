import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jax.experimental.optimizers import sgd
from openfermion.ops.operators.ising_operator import IsingOperator
from sympy import Symbol, adjoint
from sympy2jax import sympy2jax
from zquantum.core.circuits import RX, Circuit
from zquantum.core.estimation import calculate_exact_expectation_values
from zquantum.core.interfaces.estimation import EstimationTask
from zquantum.core.symbolic_simulator import SymbolicSimulator
from zquantum.core.wavefunction import Wavefunction

op = np.array([[1.0, 0.0], [0.0, -1.0]])

sim = SymbolicSimulator()

theta = 2 * Symbol("theta")
test_circuit = Circuit()
gate = RX(theta)(0)

test_circuit += gate

est_task = EstimationTask(IsingOperator("Z0"), test_circuit, 10000)

wf = sim.get_wavefunction(test_circuit)._amplitude_vector

# expectation_expression = (adjoint(wf) @ op @ wf)[0]

# f, params = sympy2jax(expectation_expression,
# list(expectation_expression.free_symbols))

# key = random.PRNGKey(0)
# X = random.normal(key, (10, 1))

# grads = f(X, params)

res = calculate_exact_expectation_values(sim, [est_task])

######### RANDOM JAX SNIPPET ################
# f, params = sympy2jax(theta, [theta])
# key = random.PRNGKey(0)
# X = random.normal(key, (10, 1))

# grads = f(X, params)
#############################################

breakpoint()
