import random
from typing import Callable, Optional

import numpy as np
import sympy
from overrides import overrides
from zquantum.core.interfaces.cost_function import CostFunction

from ..circuits import RX, Circuit
from ..measurement import Measurements
from ..symbolic_simulator import SymbolicSimulator
from ..utils import create_symbols_map
from .ansatz import Ansatz
from .ansatz_utils import ansatz_property
from .backend import QuantumBackend
from .optimizer import MetaOptimizer, Optimizer, optimization_result


class MockQuantumBackend(QuantumBackend):

    supports_batching = False

    def __init__(self):
        super().__init__()
        self._simulator = SymbolicSimulator()

    def run_circuit_and_measure(
        self, circuit: Circuit, n_samples: int, **kwargs
    ) -> Measurements:
        super(MockQuantumBackend, self).run_circuit_and_measure(circuit, n_samples)

        return self._simulator.run_circuit_and_measure(circuit, n_samples)


class MockOptimizer(Optimizer):
    def _minimize(
        self, cost_function, initial_params: np.ndarray, keep_history: bool = False
    ):
        new_parameters = initial_params
        for i in range(len(initial_params)):
            new_parameters[i] += random.random()
        new_parameters = np.array(new_parameters)
        return optimization_result(
            opt_value=cost_function(new_parameters),
            opt_params=new_parameters,
            history=[],
        )


def mock_cost_function(parameters: np.ndarray):
    return np.sum(parameters ** 2)


class MockMetaOptimizer(MetaOptimizer):
    def __init__(
        self,
        inner_optimizer: Optimizer,
        cost_function_factory: Callable[[int], CostFunction],
        n_iters: int,
    ):
        super().__init__(inner_optimizer, cost_function_factory)
        self.n_iters = n_iters

    def minimize(
        self,
        initial_params: np.ndarray,
        keep_history: bool = False,
    ):
        for i in range(self.n_iters):
            if i != 0:
                initial_params = opt_params
            cost_function = self.cost_function_factory(i)
            opt_result = self.inner_optimizer.minimize(cost_function, initial_params)
            opt_params: np.ndarray = opt_result.opt_params

        return optimization_result(
            opt_value=opt_result.opt_value, opt_params=opt_params
        )


def mock_cost_function_factory():
    return mock_cost_function


class MockAnsatz(Ansatz):

    supports_parametrized_circuits = True
    problem_size = ansatz_property("problem_size")

    def __init__(self, number_of_layers: int, problem_size: int):
        super().__init__(number_of_layers)
        self.number_of_layers = number_of_layers
        self.problem_size = problem_size

    @property
    def number_of_qubits(self) -> int:
        return self.problem_size

    @overrides
    def _generate_circuit(self, parameters: Optional[np.ndarray] = None):
        circuit = Circuit()
        symbols = [
            sympy.Symbol(f"theta_{layer_index}")
            for layer_index in range(self._number_of_layers)
        ]
        for theta in symbols:
            for qubit_index in range(self.number_of_qubits):
                circuit += RX(theta)(qubit_index)
        if parameters is not None:
            symbols_map = create_symbols_map(symbols, parameters)
            circuit = circuit.bind(symbols_map)
        return circuit
