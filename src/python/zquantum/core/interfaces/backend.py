import warnings
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Any

import numpy as np
from openfermion import IsingOperator, SymbolicOperator
from overrides import overrides
from pyquil.wavefunction import Wavefunction

from ..bitstring_distribution import (
    BitstringDistribution,
    create_bitstring_distribution_from_probability_distribution,
)
from ..circuit import CircuitConnectivity
from ..measurement import ExpectationValues, Measurements, expectation_values_to_real
from ..openfermion import change_operator_type, get_expectation_value
from ..wip.circuits._compatibility import AnyCircuit


class QuantumBackend(ABC):
    """
    Interface for implementing different quantum backends.

    Args:
        n_samples (int): number of times a circuit should be sampled.

    """

    supports_batching = False
    batch_size = None

    def __init__(self, n_samples: Optional[int] = None):
        if n_samples is not None:
            warnings.warn(
                """The n_samples attribute is deprecated. In future releases,
                n_samples will need to be passed as an argument to
                run_circuit_and_measure or run_circuitset_and_measure.""".replace(
                    "\n", ""
                ),
                DeprecationWarning,
            )
        self.n_samples = n_samples
        self.number_of_circuits_run = 0
        self.number_of_jobs_run = 0

        if self.supports_batching:
            assert isinstance(self.batch_size, int)
            assert self.batch_size > 0

    @abstractmethod
    def run_circuit_and_measure(
        self, circuit: AnyCircuit, n_samples: Optional[int] = None, **kwargs
    ) -> Measurements:
        """
        Method for executing the circuit and measuring the outcome.
        Args:
            circuit: quantum circuit to be executed.
            n_samples: The number of samples to collect. If None, the
                number of samples is determined by the n_samples attribute.

        Returns:
            core.measurement.Measurements: Object representing the measurements
                resulting from the circuit.
        """
        self.number_of_circuits_run += 1
        self.number_of_jobs_run += 1

        # This value is only returned so that mypy doesn't complain.
        # You can remove this workaround when we reimplement counter increments in
        # a more type-elegant way.
        return Measurements()

    def run_circuitset_and_measure(
        self,
        circuits: Sequence[AnyCircuit],
        n_samples: Optional[List[int]] = None,
        **kwargs
    ) -> List[Measurements]:
        """Run a set of circuits and measure a certain number of bitstrings.

        It may be useful to override this method for backends that support
        batching. Note that self.n_samples shots are used for each circuit.

        Args:
            circuits: The circuits to execute.
            n_samples: The number of samples to collect for each circuit. If
                None, the number of samples for each circuit is given by the
                n_samples attribute.

        Returns:
            Measurements for each circuit.
        """
        measurement_set: List[Measurements]

        if not self.supports_batching:
            measurement_set = []
            if n_samples is not None:
                for circuit, n_samples_for_circuit in zip(circuits, n_samples):
                    measurement_set.append(
                        self.run_circuit_and_measure(
                            circuit, n_samples=n_samples_for_circuit, **kwargs
                        )
                    )
            else:
                for circuit in circuits:
                    measurement_set.append(
                        self.run_circuit_and_measure(circuit, **kwargs)
                    )

            return measurement_set
        else:
            self.number_of_circuits_run += len(circuits)
            if isinstance(self.batch_size, int):
                self.number_of_jobs_run += int(np.ceil(len(circuits) / self.batch_size))

            # This value is only returned so that mypy doesn't complain.
            # You can remove this workaround when we reimplement counter increments in
            # a more type-elegant way.
            measurement_set = []
            return measurement_set

    def get_expectation_values(
        self, circuit: AnyCircuit, operator: SymbolicOperator, **kwargs
    ) -> ExpectationValues:
        """
        Executes the circuit and calculates the expectation values for given operator.

        Args:
            circuit: quantum circuit to be executed.
            operator: Operator for which we calculate the expectation value.

        Returns:
            Object representing expectation values for given operator.
        """
        measurements = self.run_circuit_and_measure(circuit)
        operator = change_operator_type(operator, IsingOperator)
        expectation_values = measurements.get_expectation_values(operator)
        expectation_values = expectation_values_to_real(expectation_values)
        return expectation_values

    def get_expectation_values_for_circuitset(
        self, circuits: Sequence[AnyCircuit], operator: SymbolicOperator, **kwargs
    ) -> List[ExpectationValues]:
        """Calculates the expectation values for given operator, based on the exact
        quantum state produced by a set of circuits.

        Args:
            circuits: quantum circuits to be executed.
            operator: Operator for which we calculate the expectation value.

        Returns:
            List of objects representing expectation values for given operator.
        """
        if not self.supports_batching:
            expectation_values_set = []
            for circuit in circuits:
                expectation_values_set.append(
                    self.get_expectation_values(circuit, operator, **kwargs)
                )
            return expectation_values_set
        else:
            measurements_set = self.run_circuitset_and_measure(circuits)

            expectation_values_set = []
            for measurements in measurements_set:
                expectation_values = measurements.get_expectation_values(operator)
                expectation_values = expectation_values_to_real(expectation_values)
                expectation_values_set.append(expectation_values)

            return expectation_values_set

    def get_bitstring_distribution(
        self, circuit: AnyCircuit, **kwargs
    ) -> BitstringDistribution:
        """Calculates a bitstring distribution.

        Args:
            circuit: quantum circuit to be executed.

        Returns:
            Probability distribution of getting specific bistrings.

        """
        # Get the expectation values
        measurements = self.run_circuit_and_measure(circuit, **kwargs)
        return measurements.get_distribution()


class QuantumSimulator(QuantumBackend):
    @abstractmethod
    def __init__(
        self,
        n_samples: Optional[int] = None,
        noise_model: Optional[Any] = None,
        device_connectivity: Optional[CircuitConnectivity] = None,
    ):
        super().__init__(n_samples)

    @abstractmethod
    def get_wavefunction(self, circuit: AnyCircuit, **kwargs) -> Wavefunction:
        """Returns a wavefunction representing quantum state produced by a circuit

        Args:
            circuit: quantum circuit to be executed.
        """
        self.number_of_circuits_run += 1
        self.number_of_jobs_run += 1

    @overrides
    def get_expectation_values(
        self, circuit: AnyCircuit, operator: SymbolicOperator, **kwargs
    ) -> ExpectationValues:
        """Run a circuit and measure the expectation values of given operator.

        Note:
            The number of bitstrings measured is derived
            from `self.n_samples` - if `self.n_samples == None`, then this will use
            `self.get_exact_expectation_values`

        Args:
            circuit: the circuit to prepare the state
            qubit_operator: the operator to measure

        Returns:
            The expectation values of each term in the operator
        """
        if self.n_samples is None:
            return self.get_exact_expectation_values(circuit, operator, **kwargs)
        else:
            return super().get_expectation_values(circuit, operator, **kwargs)

    @overrides
    def get_expectation_values_for_circuitset(
        self, circuits: Sequence[AnyCircuit], operator: SymbolicOperator, **kwargs
    ) -> List[ExpectationValues]:
        """Calculates the expectation values for given operator, based on the exact
        quantum state produced by a set of circuits.

        Args:
            circuits: quantum circuits to be executed.
            operator: Operator for which we calculate the expectation value.

        Returns:
            Expectation values for given operator.
        """
        if not self.supports_batching:
            expectation_values_set = []
            for circuit in circuits:
                expectation_values_set.append(
                    self.get_expectation_values(circuit, operator, **kwargs)
                )
            return expectation_values_set
        else:
            if self.n_samples is None:
                warnings.warn(
                    "When using exact simulation, batching circuits is not supported "
                    "by default."
                )
                return [
                    self.get_exact_expectation_values(circuit, operator, **kwargs)
                    for circuit in circuits
                ]
            else:
                measurements_set = self.run_circuitset_and_measure(circuits)

                expectation_values_set = []
                for measurements in measurements_set:
                    expectation_values = measurements.get_expectation_values(operator)
                    expectation_values = expectation_values_to_real(expectation_values)
                    expectation_values_set.append(expectation_values)

                return expectation_values_set

    def get_exact_expectation_values(
        self, circuit: AnyCircuit, operator: SymbolicOperator, **kwargs
    ) -> ExpectationValues:
        """Calculates the expectation values for given operator, based on the exact
        quantum state produced by circuit.

        Args:
            circuit: quantum circuit to be executed.
            operator: Operator for which we calculate the expectation value.

        Returns:
            Expectation values for given operator.
        """
        wavefunction = self.get_wavefunction(circuit)
        expectation_values = ExpectationValues(
            np.array([get_expectation_value(term, wavefunction) for term in operator])
        )
        expectation_values = expectation_values_to_real(expectation_values)
        return expectation_values

    def get_bitstring_distribution(
        self, circuit: AnyCircuit, **kwargs
    ) -> BitstringDistribution:
        """Calculates a bitstring distribution.

        Args:
            circuit: quantum circuit to be executed.

        Returns:
            Probability distribution of getting specific bistrings.
        """
        if self.n_samples is None:
            wavefunction = self.get_wavefunction(circuit, **kwargs)
            return create_bitstring_distribution_from_probability_distribution(
                wavefunction.probabilities()
            )
        else:
            # Get the expectation values
            measurements = self.run_circuit_and_measure(circuit, **kwargs)
            return measurements.get_distribution()
