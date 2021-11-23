from typing import Any, List, Sequence

from backend import QuantumBackend
from ..circuits import Circuit
from ..measurement import Measurements

""" I have assumed that get_bitstring_distribution and get_measurement_outcome_distribution
call run_circuit_and_measure or run_circuitset and measure. This might not work for some
implementations.
"""
OVERRIDDEN_METHOD_NAMES = ["run_circuit_and_measure", "run_circuitset_and_measure"]


class MeasurementTrackingBackend(QuantumBackend):
    """A wrapper class for a backend that tracks all measurements. The measurements
    are stored in the raw_circuit_data variable as a list of measurement objects.
    """

    def __init__(self, inner_backend: QuantumBackend):
        self.inner_backend = inner_backend
        self.raw_measurement_data = []
        self.implemented_circuits = []

        for attr_name in dir(inner_backend):
            if not attr_name in OVERRIDDEN_METHOD_NAMES:
                setattr(self, attr_name, getattr(inner_backend, attr_name))

    def run_circuit_and_measure(self, circuit: Circuit, n_samples: int) -> Measurements:
        """Method for executing the circuit and measuring the outcome.

        Args:
            circuit: quantum circuit to be executed.
            n_samples: The number of samples to collect.
        """
        measurements = self.inner_backend.run_circuit_and_measure(circuit, n_samples)
        self.raw_measurement_data[self._get_circuit_index(circuit)] += [measurements]
        return measurements

    def run_circuitset_and_measure(
        self, circuits: Sequence[Circuit], n_samples: List[int]
    ) -> List[Measurements]:
        """Run a set of circuits and measure a certain number of bitstrings.

        Args:
            circuits: The circuits to execute.
            n_samples: The number of samples to collect for each circuit.
        """
        measurements = self.inner_backend.run_circuitset_and_measure(
            circuits, n_samples
        )
        for i in range(len(circuits)):
            self.raw_measurement_data[self._get_circuit_index(circuits[i])] += [
                measurements[i]
            ]
        return measurements

    def _get_circuit_index(self, circuit: Circuit):
        """Returns the index of the given circuit in implemented_circuits. Index also
        corresponds to the index of the measurement data for given circuit in
        raw_measurement_data.

        Args:
            circuit: Circuit which we find the index of.
        """
        for i in range(len(self.implemented_circuits)):
            if self.implemented_circuits == circuit:
                return i
        self.implemented_circuits += [circuit]
        self.raw_measurement_data += [[]]
        return len(self.implemented_circuits)
