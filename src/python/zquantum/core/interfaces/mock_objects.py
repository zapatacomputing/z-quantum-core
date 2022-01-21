import random
from collections import defaultdict
from typing import Callable, Dict, List, Optional, cast

import numpy as np
import sympy
from overrides import overrides
from zquantum.core.history.recorder import recorder as _recorder
from zquantum.core.interfaces.cost_function import CostFunction

from ..circuits import RX, Circuit
from ..measurement import Measurements
from ..symbolic_simulator import SymbolicSimulator
from ..typing import AnyRecorder, RecorderFactory
from ..utils import create_symbols_map
from .ansatz import Ansatz
from .ansatz_utils import ansatz_property
from .backend import QuantumBackend
from .optimizer import (
    NestedOptimizer,
    Optimizer,
    construct_history_info,
    extend_histories,
    optimization_result,
)


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
            nfev=1,
            **construct_history_info(cost_function, keep_history),
        )


def mock_cost_function(parameters: np.ndarray):
    return np.sum(parameters ** 2)


class MockNestedOptimizer(NestedOptimizer):
    """
    As most mock objects this implementation does not make much sense in itself,
    however it's an example of how a NestedOptimizer could be implemented.

    """

    @property
    def inner_optimizer(self) -> Optimizer:
        return self._inner_optimizer

    @property
    def recorder(self) -> RecorderFactory:
        return self._recorder

    def __init__(
        self,
        inner_optimizer: Optimizer,
        n_iters: int,
        recorder: RecorderFactory = _recorder,
    ):
        self._inner_optimizer = inner_optimizer
        self.n_iters = n_iters
        self._recorder = recorder

    def _minimize(
        self,
        cost_function_factory: Callable[[int], CostFunction],
        initial_params: np.ndarray,
        keep_history: bool = False,
    ):
        histories: Dict[str, List] = defaultdict(list)
        histories["history"] = []
        nfev = 0
        current_params = initial_params
        for i in range(self.n_iters):
            if i != 0:
                # Increase the length of params every iteration
                # and repeats optimization with the longer params vector.
                current_params = np.append(current_params, 1)

            # Cost function changes with every iteration of NestedOptimizer
            # because it's dependent on iteration number
            cost_function = cost_function_factory(i)
            if keep_history:
                cost_function = self.recorder(cost_function)
            opt_result = self.inner_optimizer.minimize(cost_function, initial_params)
            nfev += opt_result.nfev
            current_params = opt_result.opt_params
            if keep_history:
                histories = extend_histories(
                    cast(AnyRecorder, cost_function), histories
                )
        return optimization_result(
            opt_value=opt_result.opt_value,
            opt_params=current_params,
            nit=self.n_iters,
            nfev=nfev,
            **histories,
        )


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
