from abc import ABC, abstractmethod
from ..bitstring_distribution import (
    BitstringDistribution,
    create_bitstring_distribution_from_probability_distribution,
)
from ..circuit import Circuit, CircuitConnectivity
from ..measurement import ExpectationValues, Measurements, expectation_values_to_real
from ..openfermion import expectation
from typing import Optional, List, Tuple, Iterable
from openfermion import IsingOperator, SymbolicOperator
from pyquil.wavefunction import Wavefunction
from overrides import overrides
import warnings


class QuantumBackend(ABC):
    """
    Interface for implementing different quantum backends.

    Args:
        n_samples (int): number of times a circuit should be sampled.

    """

    supports_batching = False

    def __init__(self, n_samples: Optional[int] = None):
        self.n_samples = n_samples
        self.number_of_circuits_run = 0
        self.number_of_jobs_run = 0

    @abstractmethod
    def run_circuit_and_measure(self, circuit: Circuit, **kwargs) -> Measurements:
        """
        Method for executing the circuit and measuring the outcome.
        Args:
            circuit (core.circuit.Circuit): quantum circuit to be executed.
        Returns:
            core.measurement.Measurements: object representing the measurements resulting from the circuit
        """
        self.number_of_circuits_run += 1
        self.number_of_jobs_run += 1

    def run_circuitset_and_measure(
        self, circuit_set: Iterable[Circuit], **kwargs
    ) -> List[Measurements]:
        """Run a set of circuits and measure a certain number of bitstrings.

        It may be useful to override this method for backends that support
        batching. Note that self.n_samples shots are used for each circuit.

        Args:
            circuit_set: The circuits to execute.

        Returns:
            Measurements for each circuit.
        """
        if not self.supports_batching:
            measurement_set = []
            for circuit in circuit_set:
                measurement_set.append(self.run_circuit_and_measure(circuit), **kwargs)
            return measurement_set
        else:
            self.number_of_circuits_run += len(circuit_set)
            self.number_of_jobs_run += 1

    def get_expectation_values(
        self, circuit: Circuit, operator: IsingOperator, **kwargs
    ) -> ExpectationValues:
        """
        Executes the circuit and calculates the expectation values for given operator.

        Args:
            circuit (core.circuit.Circuit): quantum circuit to be executed.
            operator(openfermion.IsingOperator): Operator for which we calculate the expectation value.

        Returns:
            ExpectationValues: object representing expectation values for given operator.
        """
        measurements = self.run_circuit_and_measure(circuit)
        expectation_values = measurements.get_expectation_values(operator)
        expectation_values = expectation_values_to_real(expectation_values)
        return expectation_values

    def get_expectation_values_for_circuitset(
        self, circuitset: List[Circuit], operator: IsingOperator, **kwargs
    ) -> List[ExpectationValues]:
        """
        Calculates the expectation values for given operator, based on the exact quantum state
        produced by a set of circuits.

        Args:
            circuitset ([core.circuit.Circuit]): quantum circuits to be executed.
            operator(openfermion.IsingOperator): Operator for which we calculate the expectation value.

        Returns:
            List[ExpectationValues]: list of objects representing expectation values for given operator.
        """
        if not self.supports_batching:
            expectation_values_set = []
            for circuit in circuitset:
                expectation_values_set.append(
                    self.get_expectation_values(circuit, operator, **kwargs)
                )
            return expectation_values_set
        else:
            measurements_set = self.run_circuitset_and_measure(circuitset)

            expectation_values_set = []
            for measurements in measurements_set:
                expectation_values = measurements.get_expectation_values(operator)
                expectation_values = expectation_values_to_real(expectation_values)
                expectation_values_set.append(expectation_values)

            return expectation_values_set

    def get_bitstring_distribution(
        self, circuit: Circuit, **kwargs
    ) -> BitstringDistribution:
        """
        Calculates a bitstring distribution.

        Args:
            circuit (core.circuit.Circuit): quantum circuit to be executed.

        Returns:
            BitstringDistribution: object representing the probabilities of getting specific bistrings.

        """
        # Get the expectation values
        measurements = self.run_circuit_and_measure(circuit, **kwargs)
        return measurements.get_distribution()


class QuantumSimulator(QuantumBackend):
    @abstractmethod
    def __init__(
        self,
        n_samples: Optional[int] = None,
        noise_model: Optional = None,
        device_connectivity: Optional[CircuitConnectivity] = None,
    ):
        super().__init__(n_samples)

    @abstractmethod
    def get_wavefunction(self, circuit: Circuit, **kwargs) -> Wavefunction:
        """
        Returns a wavefunction representing quantum state produced by a circuit

        Args:
            circuit (core.circuit.Circuit): quantum circuit to be executed.

        Returns:
            pyquil.Wafefunction: wavefunction object.

        """
        self.number_of_circuits_run += 1
        self.number_of_jobs_run += 1

    @overrides
    def get_expectation_values(self, circuit, operator: SymbolicOperator, **kwargs):
        """Run a circuit and measure the expectation values with respect to a
        given operator. Note: the number of bitstrings measured is derived
        from self.n_samples - if self.n_samples = None, then this will use
        self.get_exact_expectation_values

        Args:
            circuit (zquantum.core.circuit.Circuit): the circuit to prepare the state
            qubit_operator (openfermion.SymbolicOperator): the operator to measure
        Returns:
            zquantum.core.measurement.ExpectationValues: the expectation values
                of each term in the operator
        """
        if self.n_samples == None:
            return self.get_exact_expectation_values(circuit, operator, **kwargs)
        else:
            return super().get_expectation_values(circuit, operator, **kwargs)

    @overrides
    def get_expectation_values_for_circuitset(
        self, circuitset: List[Circuit], operator: SymbolicOperator, **kwargs
    ) -> List[ExpectationValues]:
        """
        Calculates the expectation values for given operator, based on the exact quantum state
        produced by a set of circuits.

        Args:
            circuitset ([core.circuit.Circuit]): quantum circuits to be executed.
            operator(openfermion.SymbolicOperator): Operator for which we calculate the expectation value.

        Returns:
            List[ExpectationValues]: list of objects representing expectation values for given operator.
        """
        if not self.supports_batching:
            expectation_values_set = []
            for circuit in circuitset:
                expectation_values_set.append(
                    self.get_expectation_values(circuit, operator, **kwargs)
                )
            return expectation_values_set
        else:
            if self.n_samples is None:
                warnings.warn(
                    "When using exact simulation, batching circuits is not supported by default."
                )
                return [
                    self.get_exact_expectation_values(circuit, operator ** kwargs)
                    for circuit in circuitset
                ]
            else:
                measurements_set = self.run_circuitset_and_measure(circuitset)

                expectation_values_set = []
                for measurements in measurements_set:
                    expectation_values = measurements.get_expectation_values(operator)
                    expectation_values = expectation_values_to_real(expectation_values)
                    expectation_values_set.append(expectation_values)

                return expectation_values_set

    @abstractmethod
    def get_exact_expectation_values(
        self, circuit: Circuit, operator: SymbolicOperator, **kwargs
    ) -> ExpectationValues:
        """
        Calculates the expectation values for given operator, based on the exact quantum state produced by circuit.

        Args:
            circuit (core.circuit.Circuit): quantum circuit to be executed.
            operator(openfermion): Operator for which we calculate the expectation value.

        Returns:
            ExpectationValues: object representing expectation values for given operator.
        """
        wavefunction = self.get_wavefunction(circuit)
        return ExpectationValues(expectation(operator, wavefunction))

    def get_bitstring_distribution(
        self, circuit: Circuit, **kwargs
    ) -> BitstringDistribution:
        """
        Calculates a bitstring distribution.

        Args:
            circuit (core.circuit.Circuit): quantum circuit to be executed.

        Returns:
            BitstringDistribution: object representing the probabilities of getting specific bistrings.

        """
        if self.n_samples == None:
            wavefunction = self.get_wavefunction(circuit, **kwargs)
            return create_bitstring_distribution_from_probability_distribution(
                wavefunction.probabilities()
            )
        else:
            # Get the expectation values
            measurements = self.run_circuit_and_measure(circuit, **kwargs)
            return measurements.get_distribution()
