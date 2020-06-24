from .ansatz import Ansatz
from .ansatz_utils import ansatz_property
from .backend import QuantumSimulator
from .optimizer import Optimizer
from .cost_function import CostFunction
from ..measurement import ExpectationValues, Measurements
from ..circuit import Circuit
from ..utils import create_symbols_map
import random
from scipy.optimize import OptimizeResult
import numpy as np
from pyquil import Program
from pyquil.gates import RX
import sympy
from overrides import overrides
from typing import Optional
from openfermion import SymbolicOperator


class MockQuantumSimulator(QuantumSimulator):
    def __init__(self, n_samples: Optional[int] = None):
        self.n_samples = n_samples

    def run_circuit_and_measure(self, circuit: Circuit, **kwargs):
        n_qubits = len(circuit.qubits)
        measurements = Measurements()
        for _ in range(self.n_samples):
            measurements.bitstrings += [
                tuple([random.randint(0, 1) for j in range(n_qubits)])
            ]
        return measurements

    def get_expectation_values(
        self, circuit: Circuit, operator: SymbolicOperator, **kwargs
    ):
        n_qubits = len(circuit.qubits)
        values = [random.random() for i in range(n_qubits)]
        return ExpectationValues(values)

    def get_exact_expectation_values(
        self, circuit: Circuit, operator: SymbolicOperator, **kwargs
    ):
        return self.get_expectation_values(circuit)

    def get_wavefunction(self, circuit):
        raise NotImplementedError


class MockOptimizer(Optimizer):
    def minimize(
        self, cost_function: CostFunction, initial_params: np.ndarray, **kwargs
    ):
        result = OptimizeResult()
        new_parameters = initial_params
        for i in range(len(initial_params)):
            new_parameters[i] += random.random()
        new_parameters = np.array(new_parameters)
        result.opt_value = cost_function.evaluate(new_parameters)
        result["history"] = cost_function.evaluations_history
        result.opt_params = new_parameters
        return result


class MockCostFunction(CostFunction):
    def _evaluate(self, parameters: np.ndarray):
        return np.sum(np.power(parameters, 2))

    def get_gradient(self, parameters: np.ndarray):
        if self.gradient_type == "custom":
            return np.asarray(2 * parameters)
        else:
            return self.get_gradients_finite_difference(parameters)


class MockAnsatz(Ansatz):

    supports_parametrized_circuits = True
    n_qubits = ansatz_property("n_qubits")

    def __init__(self, n_layers: int, n_qubits: int):
        super().__init__(n_layers)
        self.n_layers = n_layers
        self.n_qubits = n_qubits

    @overrides
    def _generate_circuit(self, parameters: Optional[np.ndarray] = None):
        circuit = Circuit()
        for theta in self.get_symbols():
            for qubit_index in range(self.n_qubits):
                circuit += Circuit(Program(RX(theta, qubit_index)))
        if parameters is not None:
            symbols_map = create_symbols_map(self.get_symbols(), parameters)
            circuit = circuit.evaluate(symbols_map)
        return circuit

    @overrides
    def get_symbols(self):
        return [
            sympy.Symbol(f"theta_{layer_index}")
            for layer_index in range(self._n_layers)
        ]
