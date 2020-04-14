from abc import ABC, abstractmethod
from ..bitstring_distribution import BitstringDistribution, create_bitstring_distribution_from_probability_distribution, create_bitstring_distribution_from_measurements
from ..circuit import Circuit
from ..measurement import ExpectationValues
from typing import Optional, List, Tuple
from openfermion import QubitOperator
from pyquil.wavefunction import Wavefunction

class QuantumBackend(ABC):
    """
    Interface for implementing different quantum backends. 
    
    Args:
        n_samples (int): number of times a circuit should be sampled.

    """
    def __init__(self, n_samples:Optional[int]=None):
        self.n_samples = n_samples

    @abstractmethod
    def run_circuit_and_measure(self, circuit:Circuit, **kwargs) -> List[Tuple]:
        """
        Method for executing the circuit and measuring the outcome.
        Args:
            circuit (core.circuit.Circuit): quantum circuit to be executed.
        """
        raise NotImplementedError

    def get_expectation_values(self, circuit:Circuit, qubit_operator:QubitOperator, **kwargs) -> ExpectationValues:
        """
        Executes the circuit and calculates the expectation values for given operator.

        Args:
            circuit (core.circuit.Circuit): quantum circuit to be executed.
            qubit_operator(openfermion): Operator for which we calculate the expectation value.

        Returns:
            ExpectationValues: object representing expectation values for given operator.
        """
        raise NotImplementedError


class QuantumSimulator(QuantumBackend):
    @abstractmethod
    def __init__(self, n_samples:Optional[int]=None):
        self.n_samples = n_samples

    @abstractmethod
    def get_wavefunction(self, circuit:Circuit, **kwargs) -> Wavefunction:
        """
        Returns a wavefunction representing quantum state produced by a circuit

        Args:
            circuit (core.circuit.Circuit): quantum circuit to be executed.
        
        Returns:
            pyquil.Wafefunction: wavefunction object.

        """

        raise NotImplementedError

    @abstractmethod
    def get_exact_expectation_values(self, circuit:Circuit, qubit_operator:QubitOperator, **kwargs) -> ExpectationValues:
        """
        Calculates the expectation values for given operator, based on the exact quantum state produced by circuit.

        Args:
            circuit (core.circuit.Circuit): quantum circuit to be executed.
            qubit_operator(openfermion): Operator for which we calculate the expectation value.

        Returns:
            ExpectationValues: object representing expectation values for given operator.
        """

        raise NotImplementedError

    def get_bitstring_distribution(self, circuit:Circuit, **kwargs) -> BitstringDistribution:
        """
        Calculates a bitstring distribution.

        Args:
            circuit (core.circuit.Circuit): quantum circuit to be executed.

        Returns:
            BitstringDistribution: object representing the probabilities of getting specific bistrings.

        """
        if self.n_samples == None:
            wavefunction = self.get_wavefunction(circuit, **kwargs)
            return create_bitstring_distribution_from_probability_distribution(wavefunction.probabilities())
        else:
            # Get the expectation values
            measurements = self.run_circuit_and_measure(circuit, **kwargs)
            return create_bitstring_distribution_from_measurements(measurements)
