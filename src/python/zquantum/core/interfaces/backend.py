from abc import ABC, abstractmethod
from itertools import groupby
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from openfermion import IsingOperator, QubitOperator, SymbolicOperator
from zquantum.core.wavefunction import Wavefunction

from ..bitstring_distribution import (
    BitstringDistribution,
    create_bitstring_distribution_from_probability_distribution,
)
from ..circuits import Circuit, GateOperation, Operation
from ..circuits.layouts import CircuitConnectivity
from ..measurement import ExpectationValues, Measurements, expectation_values_to_real
from ..openfermion import change_operator_type, get_expectation_value


class QuantumBackend(ABC):
    """Interface for implementing different quantum backends.

    Attributes:
        supports_batching: boolean flag indicating whether given backend
            supports batching circuits.
        batch_size: number of circuit runs in a single batch.
            If `supports_batching` is true should be a positive integer.
        number_of_circuits_run: number of circuits executed by this backend
        number_of_jobs_run: number of jobs executed by this backend. Will be different
            from `number_of_circuits_run` if batches are used.
    """

    supports_batching: bool = False
    batch_size: Optional[int] = None

    def __init__(self):
        self.number_of_circuits_run = 0
        self.number_of_jobs_run = 0

        if self.supports_batching:
            assert isinstance(self.batch_size, int)
            assert self.batch_size > 0

    @abstractmethod
    def run_circuit_and_measure(self, circuit: Circuit, n_samples: int) -> Measurements:
        """
        Method for executing the circuit and measuring the outcome.
        Args:
            circuit: quantum circuit to be executed.
            n_samples: The number of samples to collect.
        """
        assert isinstance(n_samples, int) and n_samples > 0
        self.number_of_circuits_run += 1
        self.number_of_jobs_run += 1

        # NOTE: This value is only returned so that mypy doesn't complain.
        # You can remove this workaround when we reimplement counter increments in
        # a more type-elegant way.
        return Measurements()

    def run_circuitset_and_measure(
        self, circuits: Sequence[Circuit], n_samples: List[int]
    ) -> List[Measurements]:
        """Run a set of circuits and measure a certain number of bitstrings.

        It may be useful to override this method for backends that support
        batching.

        Args:
            circuits: The circuits to execute.
            n_samples: The number of samples to collect for each circuit.
        """
        measurement_set: List[Measurements]

        if not self.supports_batching:
            measurement_set = []
            for circuit, n_samples_for_circuit in zip(circuits, n_samples):
                measurement_set.append(
                    self.run_circuit_and_measure(
                        circuit, n_samples=n_samples_for_circuit
                    )
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

    def get_bitstring_distribution(
        self, circuit: Circuit, n_samples: int
    ) -> BitstringDistribution:
        """Calculates a bitstring distribution.

        Args:
            circuit: quantum circuit to be executed.

        Returns:
            Probability distribution of getting specific bistrings.

        """
        # Get the expectation values
        measurements = self.run_circuit_and_measure(circuit, n_samples)
        return measurements.get_distribution()


def split_circuit(
    circuit: Circuit, predicate: Callable[[Operation], bool]
) -> Iterable[Tuple[bool, Circuit]]:
    """Split circuit into subcircuits for which predicate on all operation is constant.

    Args:
        circuit: a circuit to be split
        predicate: function assigning boolean value to each operation, its values
          are used for grouping operations belonging to the same subcircuits.
    Returns:
        An iterable of tuples of the form (x, subcircuit) s.t.:
        - predicate(operation) == x for every operation in subcircuit.operations
        - for two consecutive tuples (x1, subcircuit1), (x2, subcircuit2)
          x1 != x2 (i.e. consecutive chunks differ in the predicate value),
        - operations in subcircuits follow the same order as in original circuit
        - all subcircuits have the same number of qubits equal to `circuit.n_qubits`.
    """
    n_qubits = circuit.n_qubits
    for predicate_value, operations in groupby(circuit.operations, predicate):
        yield predicate_value, Circuit(operations, n_qubits=n_qubits)


class QuantumSimulator(QuantumBackend):
    @abstractmethod
    def __init__(
        self,
        noise_model: Optional[Any] = None,
        device_connectivity: Optional[CircuitConnectivity] = None,
    ):
        super().__init__()
        self.noise_model = noise_model
        self.device_connectivity = device_connectivity

    @abstractmethod
    def _get_wavefunction_from_native_circuit(
        self, circuit: Circuit, initial_state
    ) -> Wavefunction:
        """Get wavefunction from circuit comprising only natively-supported operations.

        Args:
            circuit: circuit to simulate. Implementers of this function might assume
              that this circuit comprises only natively-supported operations as decided
              by self._is_supported predicate.
            initial_state: amplitudes of the initial state.
        Returns:
            Wavefunction object encoding amplitudes of final state.
        """

    def is_natively_supported(self, operation: Operation) -> bool:
        """Determine if given operation is natively supported by this Simulator.

        This method can be as a predicate in split_circuit function.
        """
        return isinstance(operation, GateOperation)

    def get_wavefunction(self, circuit: Circuit, initial_state=None) -> Wavefunction:
        """Returns a wavefunction representing quantum state produced by a circuit

        Args:
            circuit: quantum circuit to be executed.
            initial_state: a state from which the simulation starts.
              If not provided, the default |0...0> is used.
        """
        if initial_state is None:
            state = np.zeros(2 ** circuit.n_qubits)
            state[0] = 1
        else:
            state = initial_state

        for is_supported, subcircuit in split_circuit(
            circuit, self.is_natively_supported
        ):
            # Native subcircuits are passed through to the underlying simulator.
            # They also count towards number of circuits and number of jobs run.
            if is_supported:
                self.number_of_circuits_run += 1
                self.number_of_jobs_run += 1
                state = self._get_wavefunction_from_native_circuit(subcircuit, state)
            else:
                for operation in subcircuit.operations:
                    state = operation.apply(state)

        return Wavefunction(state)

    def get_exact_expectation_values(
        self, circuit: Circuit, operator: SymbolicOperator
    ) -> ExpectationValues:
        """Calculates the expectation values for given operator, based on the exact
        quantum state produced by circuit.

        Args:
            circuit: quantum circuit to be executed.
            operator: Operator for which we calculate the expectation value.
        """
        wavefunction = self.get_wavefunction(circuit)
        if isinstance(operator, IsingOperator):
            operator = change_operator_type(operator, QubitOperator)
        expectation_values = ExpectationValues(
            np.array([get_expectation_value(term, wavefunction) for term in operator])
        )
        expectation_values = expectation_values_to_real(expectation_values)
        return expectation_values

    def get_bitstring_distribution(
        self, circuit: Circuit, n_samples: Optional[int] = None
    ) -> BitstringDistribution:
        """Calculates a bitstring distribution.

        Args:
            circuit: quantum circuit to be executed.

        Returns:
            Probability distribution of getting specific bistrings.
        """
        if n_samples is None:
            wavefunction = self.get_wavefunction(circuit)
            return create_bitstring_distribution_from_probability_distribution(
                wavefunction.probabilities()
            )
        else:
            # Get the expectation values
            measurements = self.run_circuit_and_measure(circuit, n_samples)
            return measurements.get_distribution()


def _flip_bits(n, num_bits):
    return int(bin(n)[2:].zfill(num_bits)[::-1], 2)


def flip_wavefunction(wavefunction: Wavefunction):
    number_of_states = len(wavefunction.amplitudes)
    ordering = [
        _flip_bits(n, number_of_states.bit_length() - 1)
        for n in range(number_of_states)
    ]
    flipped_amplitudes = [wavefunction.amplitudes[i] for i in ordering]
    return Wavefunction(np.array(flipped_amplitudes))
