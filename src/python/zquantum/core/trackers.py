from typing import List, Dict, Sequence, Optional
import json
from time import localtime, asctime
from itertools import count
from os import rename
import pdb

from .interfaces.backend import QuantumBackend, QuantumSimulator
from .circuits import Circuit, to_dict
from .measurement import Measurements
from .utils import SCHEMA_VERSION
from .bitstring_distribution import BitstringDistribution
from .distribution import MeasurementOutcomeDistribution


class MeasurementTrackingBackend(QuantumBackend):
    """A wrapper class for a backend that tracks all measurements. The measurements
    are stored in the raw_circuit_data variable as a list of measurement objects.

    To enable tracking, simply put the key-value pair "track_measurements" : True
    in your backend_specs. You can also specify whether or not to keep all the
    bitstrings measured by adding the key-value pair "record_bitstrings" : True.
    However keeping all the bitstrings from an experiment may create a very large
    amount of data.

    In a workflow, to catch the data provided by MeasurementTrackingBackend you must
    put a json file named "raw_measurement_data" to keep the data in.
    """

    id_iter = count()

    def __init__(
        self, inner_backend: QuantumBackend, record_bitstrings: Optional[bool] = False
    ):
        """Create a wrapper backend around inner_backend that keeps track of all the
        measurement data collected during an experiment.

        Args:
            inner_backend (QuantumBackend): Backend for which measurements are recorded
            record_bitstrings (bool, optional): Record every measured bitstring. May
                lead to large amounts of stored data. Defaults to False.
        """
        super().__init__()
        self.id: int = next(MeasurementTrackingBackend.id_iter)
        self.record_bitstrings: Optional[bool] = record_bitstrings
        self.inner_backend: QuantumBackend = inner_backend
        self.raw_measurement_data: List[Dict] = []
        self.type: str = inner_backend.__class__.__name__
        self.timestamp: str = asctime(localtime()).replace(" ", "-")
        self.file_name = (
            self.type + "-" + str(self.id) + "-usage-data-" + self.timestamp + ".json"
        )

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
        """Append data from a measurement to self.raw_measurement_data.

        Args:
            circuit (Circuit): Implemented circuit.
            measurement (Measurements): Implemented measurement.
        """
        raw_data_dict = {
            "device": self.type,
            "circuit": to_dict(circuit),
            "counts": measurement.get_counts(),
            "number_of_gates": len(circuit.operations),
            "number_of_shots": len(measurement.bitstrings),
        }
        if self.record_bitstrings:
            raw_data_dict["bitstrings"] = [
                list(map(int, list(bitstring))) for bitstring in measurement.bitstrings
            ]
        self.raw_measurement_data.append(raw_data_dict)

    def save_raw_measurement_data(self) -> None:
        with open(self.file_name, "w+") as f:
            data = {
                "schema": SCHEMA_VERSION + "-raw-measurement-data",
                "raw-measurement-data": self.raw_measurement_data,
            }
            f.write(json.dumps(data))
        self.raw_measurement_data = []

    def rename_raw_measurement_data_file(self) -> None:
        rename(self.file_name, "raw_measurement_data.json")
