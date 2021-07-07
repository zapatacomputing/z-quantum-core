import random
from typing import Optional

import numpy as np
import sympy
from openfermion import SymbolicOperator
from overrides import overrides
from pyquil.wavefunction import Wavefunction

from ..circuits import RX, Circuit
from ..measurement import ExpectationValues, Measurements
from ..symbolic_simulator import SymbolicSimulator
from ..utils import create_symbols_map
from .ansatz import Ansatz
from .ansatz_utils import ansatz_property
from .backend import QuantumBackend, QuantumSimulator
from .optimizer import Optimizer, optimization_result


class MockQuantumBackend(QuantumBackend):

    supports_batching = False

    def __init__(self, n_samples: Optional[int] = None):
        super().__init__(n_samples)

    def run_circuit_and_measure(
        self, circuit: Circuit, n_samples: Optional[int] = None, **kwargs
    ) -> Measurements:
        super(MockQuantumBackend, self).run_circuit_and_measure(circuit, n_samples)
        if n_samples is not None:
            simulator = SymbolicSimulator(n_samples=n_samples)
        else:
            simulator = SymbolicSimulator(n_samples=self.n_samples)
        return simulator.run_circuit_and_measure(circuit, n_samples)


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
