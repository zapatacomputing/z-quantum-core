from typing import List, Sequence, Dict
import json

from interfaces.backend import QuantumBackend, QuantumSimulator
from circuits import Circuit
from measurement import Measurements
from utils import SCHEMA_VERSION
from bitstring_distribution import BitstringDistribution
from distribution import MeasurementOutcomeDistribution


class MeasurementTrackingBackend(QuantumBackend):
    """A wrapper class for a backend that tracks all measurements. The measurements
    are stored in the raw_circuit_data variable as a list of measurement objects.
    """

    def __init__(self, inner_backend: QuantumBackend):
        if isinstance(inner_backend, QuantumSimulator):
            raise TypeError("Backend tracking for simulators is not supported.")
        super().__init__()
        self.inner_backend: QuantumBackend = inner_backend
        self.raw_measurement_data: List[Dict] = []

    def run_circuit_and_measure(self, circuit: Circuit, n_samples: int) -> Measurements:
        """Method for executing the circuit and measuring the outcome.

        Args:
            circuit: quantum circuit to be executed.
            n_samples: The number of samples to collect.
        """
        measurement = self.inner_backend.run_circuit_and_measure(circuit, n_samples)
        self.record_raw_measurement_data(circuit, measurement)
        self.save_raw_measurement_data()
        return measurement

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
        for circuit, measurement in zip(circuits, measurements):
            self.record_raw_measurement_data(circuit, measurement)
        self.save_raw_measurement_data()
        return measurements

    def get_bitstring_distribution(
        self, circuit: Circuit, n_samples: int
    ) -> BitstringDistribution:
        """Calculates a bitstring distribution.

        This function is a wrapper around `get_measurement_outcome_distribution`
        needed for backward-compatibility.

        Args:
            circuit: quantum circuit to be executed.

        Returns:
            Probability distribution of getting specific bistrings.
        """
        return self.inner_backend.get_bitstring_distribution(circuit, n_samples)

    def get_measurement_outcome_distribution(
        self, circuit: Circuit, n_samples: int
    ) -> MeasurementOutcomeDistribution:
        """Calculates a measurement outcome distribution.

        Args:
            circuit: quantum circuit to be executed.

        Returns:
            Probability distribution of getting specific bistrings.

        """
        return self.inner_backend.get_measurement_outcome_distribution(
            circuit, n_samples
        )

    def record_raw_measurement_data(
        self, circuit: Circuit, measurement: Measurements
    ) -> None:
        self.raw_measurement_data.append(
            {
                "device": self.inner_backend.__name__,
                "circuit": circuit.to_dict(serialize_gate_params=True),
                "counts": measurement.get_counts(),
                "bitstrings": [
                    list(map(int, list(bitstring)))
                    for bitstring in measurement.bitstrings
                ],
                "number_of_multiqubit_gates": circuit.n_multiqubit_gates,
                "number_of_gates": len(circuit.gates),
            }
        )

    def save_raw_measurement_data(self) -> None:
        with open("backend-usage-data.json", "w") as f:
            data = {
                "schema": SCHEMA_VERSION + "-raw-measurement-data",
                "raw-measurement-data": self.raw_measurement_data,
            }
            f.write(json.dumps(data))
