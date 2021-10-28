import jax
import jax.numpy as jnp
import numpy as np
import sympy
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


def exp_of_rx_gate_numeric_openfermion(rotation):
    """
    For some reason, this returns a grad of 0 no matter what I input
    """
    test_circuit = Circuit()
    gate = RX(rotation)(0)

    test_circuit += gate

    est_task = EstimationTask(IsingOperator("Z0"), test_circuit, 10000)

    return calculate_exact_expectation_values(sim, [est_task])[0].values[0]


def exp_of_rx_gate_numeric_vanilla(rotation):
    """
    For some reason, this returns a grad of 0 no matter what I input
    """
    test_circuit = Circuit()
    gate = RX(rotation)(0)

    test_circuit += gate

    wavefunction = jnp.array(sim.get_wavefunction(test_circuit)._amplitude_vector)

    op = jnp.array([[1.0, 0.0], [0.0, -1.0]])

    return jnp.vdot(wavefunction, op @ wavefunction).real


def exp_of_rx_gate_hybrid_openfermion(rotation):
    """
    For some reason, this returns a grad of 0 no matter what I input
    """
    test_circuit = Circuit()
    gate = RX(2 * Symbol("theta"))(0)

    test_circuit += gate

    test_circuit = test_circuit.bind({"theta": rotation})

    est_task = EstimationTask(IsingOperator("Z0"), test_circuit, 10000)

    return calculate_exact_expectation_values(sim, [est_task])[0].values[0]


def exp_of_rx_gate_symbolic_openfermion():
    test_circuit = Circuit()
    gate = RX(2 * Symbol("theta"))(0)

    test_circuit += gate

    est_task = EstimationTask(IsingOperator("Z0"), test_circuit, 10000)

    return calculate_exact_expectation_values(sim, [est_task])[0].values[0]


sim = SymbolicSimulator()

expectation_function = exp_of_rx_gate_numeric_vanilla
expectation_expression = expectation_function(0.0)

key = random.PRNGKey(0)
X = jnp.arange(0.0, jnp.pi, 0.25).reshape(-1, 1)

if (
    hasattr(expectation_expression, "free_symbols")
    and expectation_expression.free_symbols != []
):
    f, params = sympy2jax(
        expectation_expression, list(expectation_expression.free_symbols)
    )

    def cost_function(vals):
        return f(vals, params)[0]

    grads = jax.grad(cost_function)

    print(f"Values: {[cost_function(jnp.array([x])).item() for x in X]}")
    print(f"Grads: {[grads(jnp.array([x])).item() for x in X]}")
else:
    grads = jax.grad(expectation_function)

    print(f"Values: {[expectation_function(jnp.array([x])).item() for x in X]}")
    print(f"Grads: {[grads(jnp.array([x])).item() for x in X]}")
