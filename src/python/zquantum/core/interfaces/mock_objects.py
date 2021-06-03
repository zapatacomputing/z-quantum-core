import random
from typing import Optional

import numpy as np
import sympy
from openfermion import SymbolicOperator
from overrides import overrides
from pyquil.wavefunction import Wavefunction

from ..circuits import RX, Circuit
from ..measurement import ExpectationValues, Measurements
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
        super(MockQuantumBackend, self).run_circuit_and_measure(circuit)
        measurements = Measurements()

        n_samples_to_measure: int
        if isinstance(n_samples, int):
            n_samples_to_measure = n_samples
        elif isinstance(self.n_samples, int):
            n_samples_to_measure = self.n_samples
        else:
            raise ValueError(
                "At least one of n_samples and self.n_samples must be an integer."
            )

        for _ in range(n_samples_to_measure):
            measurements.bitstrings += [
                tuple(random.randint(0, 1) for j in range(circuit.n_qubits))
            ]

        return measurements

    def get_wavefunction(self, circuit: Circuit):
        raise NotImplementedError

    def get_density_matrix(self, circuit: Circuit):
        raise NotImplementedError


class MockQuantumSimulator(QuantumSimulator):

    supports_batching = False

    def __init__(self, n_samples: Optional[int] = None):
        super().__init__(n_samples)

    def run_circuit_and_measure(
        self, circuit: Circuit, n_samples=None, **kwargs
    ) -> Measurements:
        super(MockQuantumSimulator, self).run_circuit_and_measure(circuit)
        measurements = Measurements()
        if n_samples is None:
            n_samples = self.n_samples
        for _ in range(n_samples):
            measurements.bitstrings += [
                tuple(random.randint(0, 1) for j in range(circuit.n_qubits))
            ]

        return measurements

    def get_expectation_values(
        self, circuit: Circuit, operator: SymbolicOperator, **kwargs
    ) -> ExpectationValues:
        if self.n_samples is None:
            self.number_of_circuits_run += 1
            self.number_of_jobs_run += 1
            constant_position = None
            n_operator: Optional[int]
            if hasattr(operator, "terms"):
                n_operator = len(operator.terms.keys())
                for index, term in enumerate(operator.terms):
                    if term == ():
                        constant_position = index
            else:
                n_operator = None
                print("WARNING: operator does not have attribute terms")
            length = n_operator if n_operator is not None else circuit.n_qubits
            values = np.asarray([2.0 * random.random() - 1.0 for i in range(length)])
            if n_operator is not None and constant_position is not None:
                values[constant_position] = operator.terms[()]
            return ExpectationValues(values)
        else:
            return super(MockQuantumSimulator, self).get_expectation_values(
                circuit, operator
            )

    def get_exact_expectation_values(
        self, circuit: Circuit, operator: SymbolicOperator, **kwargs
    ) -> ExpectationValues:
        return self.get_expectation_values(circuit, operator)

    def get_wavefunction(self, circuit: Circuit, **kwargs) -> Wavefunction:
        raise NotImplementedError


class MockOptimizer(Optimizer):
    def minimize(
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
