from .ansatz import Ansatz
from .ansatz_utils import ansatz_property
from .backend import QuantumBackend, QuantumSimulator
from .optimizer import Optimizer
from .cost_function import CostFunction
from .estimator import Estimator
from ..measurement import ExpectationValues, Measurements
from ..circuit import Circuit
from ..utils import ValueEstimate, create_symbols_map
import random
from scipy.optimize import OptimizeResult
import numpy as np
from openfermion import SymbolicOperator
from pyquil import Program
from pyquil.gates import RX, X
import sympy
from overrides import overrides
from typing import Optional


class MockQuantumBackend(QuantumBackend):
    def __init__(self, n_samples=None):
        self.n_samples = n_samples

    def run_circuit_and_measure(self, circuit, **kwargs):

        n_qubits = len(circuit.qubits)
        measurements = Measurements()
        for _ in range(self.n_samples):
            measurements.bitstrings += [
                tuple([random.randint(0, 1) for j in range(n_qubits)])
            ]
        return measurements

    def get_expectation_values(self, circuit, operator, **kwargs):
        n_qubits = len(circuit.qubits)
        values = [random.random() for i in range(n_qubits)]
        return ExpectationValues(values)

    def get_wavefunction(self, circuit):
        raise NotImplementedError

    def get_density_matrix(self, circuit):
        raise NotImplementedError


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
        try:
            n_operator = len(operator.terms.keys())
            constant_position = None
            for index, term in enumerate(operator.terms):
                if term == ():
                    constant_position = index
        except:
            n_operator = None
            constant_position = None
            print("WARNING: operator does not have attribute terms")
        if n_operator is not None:
            length = n_operator
        else:
            length = n_qubits
        values = [2.0 * random.random() - 1.0 for i in range(length)]
        if n_operator is not None and constant_position is not None:
            values[constant_position] = 1.0
        return ExpectationValues(values)

    def get_exact_expectation_values(
        self, circuit: Circuit, operator: SymbolicOperator, **kwargs
    ):
        return self.get_expectation_values(circuit, operator)

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
        return ValueEstimate(np.sum(np.power(parameters, 2)))

    def get_gradient(self, parameters: np.ndarray):
        if self.gradient_type == "custom":
            return np.asarray(2 * parameters)
        else:
            return self.get_gradients_finite_difference(parameters)


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
        for theta in self.get_symbols():
            for qubit_index in range(self.number_of_qubits):
                circuit += Circuit(Program(RX(theta, qubit_index)))
        if parameters is not None:
            symbols_map = create_symbols_map(self.get_symbols(), parameters)
            circuit = circuit.evaluate(symbols_map)
        return circuit

    @overrides
    def get_symbols(self):
        return [
            sympy.Symbol(f"theta_{layer_index}")
            for layer_index in range(self._number_of_layers)
        ]


class MockEstimator(Estimator):
    @overrides
    def get_estimated_expectation_values(
        self,
        backend: QuantumBackend,
        circuit: Circuit,
        target_operator: SymbolicOperator,
        n_samples: Optional[int],
        epsilon: Optional[float],
        delta: Optional[float],
    ) -> ExpectationValues:
        return backend.get_expectation_values(circuit, target_operator)


def mock_ansatz(parameters):
    return Circuit(Program(X(0)))
