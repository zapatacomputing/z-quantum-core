from .ansatz import Ansatz
from .backend import QuantumSimulator
from .optimizer import Optimizer
from .cost_function import CostFunction
from ..measurement import ExpectationValues, Measurements
from ..circuit import Circuit
import random
from scipy.optimize import OptimizeResult
import numpy as np
from pyquil import Program
from pyquil.gates import RX
import sympy
from overrides import overrides


class MockQuantumSimulator(QuantumSimulator):
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

    def get_exact_expectation_values(self, circuit, operator, **kwargs):
        return self.get_expectation_values(circuit)

    def get_wavefunction(self, circuit):
        raise NotImplementedError

    def get_density_matrix(self, circuit):
        raise NotImplementedError


class MockOptimizer(Optimizer):
    def minimize(self, cost_function, initial_params, **kwargs):
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
    def _evaluate(self, parameters):
        return np.sum(np.power(parameters, 2))

    def get_gradient(self, parameters):
        if self.gradient_type == "custom":
            return np.asarray(2 * parameters)
        else:
            return self.get_gradients_finite_difference(parameters)


class MockAnsatz(Ansatz):
    supported_gradient_methods = ["finite_differences", "parameter_shift_rule"]

    @overrides
    def generate_circuit(self):
        circuit = Circuit()
        for layer_index in range(self._n_layers):
            theta = sympy.Symbol("theta" + str(layer_index))
            for qubit_index in range(self._n_qubits):
                circuit += Program(RX(theta, qubit_index))
        return circuit

    @overrides
    def generate_gradient_circuits(self):
        if self.gradient_type == "finite_differences":
            return [self.get_circuit()]
        elif self.gradient_type == "parameter_shift_rule":
            circuit_plus = Circuit()
            circuit_minus = Circuit()
            for layer_index in range(self._n_layers):
                theta = sympy.Symbol("theta" + str(layer_index))
                for qubit_index in range(self._n_qubits):
                    circuit_plus += Program(RX(theta + np.pi / 2, qubit_index))
                    circuit_minus += Program(RX(theta - np.pi / 2, qubit_index))
            return [circuit_plus, circuit_minus]
        else:
            raise ValueError(
                "Gradient type: {0} not supported.".format(self.gradient_type)
            )

    @overrides
    def get_number_of_params_per_layer(self):
        return 1

    @overrides
    def get_symbols(self):
        return [
            sympy.Symbol(f"theta{layer_index}") for layer_index in range(self._n_layers)
        ]
