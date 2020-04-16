from .backend import QuantumSimulator
from .optimizer import Optimizer
from ..measurement import ExpectationValues
import random
from scipy.optimize import OptimizeResult
import numpy as np

class MockQuantumSimulator(QuantumSimulator):
    def __init__(self, n_samples=None):
        self.n_samples = n_samples

    def run_circuit_and_measure(self, circuit, **kwargs):
        n_qubits = len(circuit.qubits)
        measurements = []
        for _ in range(self.n_samples):
            measurements.append(tuple([random.randint(0,1) for j in range(n_qubits)]))
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
        result.opt_value = cost_function(new_parameters)
        result['history'] = [{'value': result.opt_value, 'params': new_parameters}]
        result.opt_params = new_parameters
        return result