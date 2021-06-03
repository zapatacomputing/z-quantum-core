from __future__ import annotations

import copy
import json
from collections import Counter
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    TextIO,
    Tuple,
    Union,
    cast,
)

import numpy as np
from openfermion.ops import IsingOperator
from pyquil.wavefunction import Wavefunction
from zquantum.core.typing import AnyPath, LoadSource

from .bitstring_distribution import BitstringDistribution
from .utils import (
    SCHEMA_VERSION,
    convert_array_to_dict,
    convert_dict_to_array,
    convert_tuples_to_bitstrings,
    sample_from_probability_distribution,
)


def save_expectation_values(
    expectation_values: ExpectationValues, filename: AnyPath
) -> None:
    """Save expectation values to a file.

    Args:
        expectation_values (ExpectationValues): the expectation values to save
        file (str or file-like object): the name of the file, or a file-like object
    """
    dictionary = expectation_values.to_dict()
    dictionary["schema"] = SCHEMA_VERSION + "-expectation_values"

    with open(filename, "w") as f:
        f.write(json.dumps(dictionary, indent=2))


def load_expectation_values(file: LoadSource) -> ExpectationValues:
    """Load an array from a file.

    Args:
        file (str or file-like object): the name of the file, or a file-like object.

    Returns:
        array (numpy.array): the array
    """

    if isinstance(file, str):
        with open(file, "r") as f:
            data = json.load(f)
    else:
        data = json.load(file)

    return ExpectationValues.from_dict(data)


def load_wavefunction(file: LoadSource) -> Wavefunction:
    """Load a qubit wavefunction from a file.

    Args:
        file (str or file-like object): the name of the file, or a file-like object.

    Returns:
        wavefunction (pyquil.wavefunction.Wavefunction): the wavefunction object
    """

    if isinstance(file, str):
        with open(file, "r") as f:
            data = json.load(f)
    else:
        data = json.load(file)

    wavefunction = Wavefunction(convert_dict_to_array(data["amplitudes"]))
    return wavefunction


def save_wavefunction(wavefunction: Wavefunction, filename: AnyPath) -> None:
    """Save a wavefunction object to a file.

    Args:
        wavefunction (pyquil.wavefunction.Wavefunction): the wavefunction object
        filename (str): the name of the file
    """

    data: Dict[str, Any] = {"schema": SCHEMA_VERSION + "-wavefunction"}
    data["amplitudes"] = convert_array_to_dict(wavefunction.amplitudes)
    with open(filename, "w") as f:
        f.write(json.dumps(data, indent=2))


class ExpectationValues:
    """A class representing expectation values of operators.
    For more context on how it is being used, please see the docstring of
    EstimateExpectationValues Protocol in interfaces/estimation.py.

    Args:
        values: The expectation values of a set of operators.
        correlations: The expectation values of pairwise products of operators.
            Contains an NxN array for each frame, where N is the number of
            operators in that frame.
        estimator_covariances: The (estimated) covariances between estimates of
            expectation values of pairs of operators. Contains an NxN array for
            each frame, where N is the number of operators in that frame. Note
            that for direct sampling, the covariance between estimates of
            expectation values is equal to the population covariance divided by
            the number of samples.

    Attributes:
        values: See Args.
        correlations: See Args.
        estimator_covariances: See Args.
    """

    def __init__(
        self,
        values: np.ndarray,
        correlations: Optional[List[np.ndarray]] = None,
        estimator_covariances: Optional[List[np.ndarray]] = None,
    ):
        self.values = values
        self.correlations = correlations
        self.estimator_covariances = estimator_covariances

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary"""

        data: Dict[str, Any] = {
            "schema": SCHEMA_VERSION + "-expectation_values",
            "frames": [],
        }  # what is "frames" for?

        data["expectation_values"] = convert_array_to_dict(self.values)

        if self.correlations:
            data["correlations"] = []
            for correlation_matrix in self.correlations:
                data["correlations"].append(convert_array_to_dict(correlation_matrix))

        if self.estimator_covariances:
            data["estimator_covariances"] = []
            for covariance_matrix in self.estimator_covariances:
                data["estimator_covariances"].append(
                    convert_array_to_dict(covariance_matrix)
                )

        return data

    @classmethod
    def from_dict(cls, dictionary: dict) -> ExpectationValues:
        """Create an ExpectationValues object from a dictionary."""

        expectation_values = convert_dict_to_array(dictionary["expectation_values"])
        correlations: Optional[List] = None
        if dictionary.get("correlations"):
            correlations = []
            for correlation_matrix in cast(Iterable, dictionary.get("correlations")):
                correlations.append(convert_dict_to_array(correlation_matrix))

        estimator_covariances: Union[List, None] = None
        if dictionary.get("estimator_covariances"):
            estimator_covariances = []
            for covariance_matrix in cast(
                Iterable, dictionary.get("estimator_covariances")
            ):
                estimator_covariances.append(convert_dict_to_array(covariance_matrix))

        return cls(expectation_values, correlations, estimator_covariances)


def sample_from_wavefunction(
    wavefunction: Wavefunction, n_samples: int
) -> List[Tuple[int, ...]]:
    """Sample bitstrings from a wavefunction.

    Args:
        wavefunction (Wavefunction): the wavefunction to sample from.
        n_samples (int): the number of samples taken.

    Returns:
        List[Tuple[int]]: A list of tuples where the each tuple is a sampled bitstring.
    """
    rng = np.random.default_rng()
    outcomes_str, probabilities_np = zip(*wavefunction.get_outcome_probs().items())
    probabilities = [
        x[0] if isinstance(x, (list, np.ndarray)) else x for x in list(probabilities_np)
    ]
    samples_ndarray = rng.choice(a=outcomes_str, size=n_samples, p=probabilities)
    samples = [tuple(int(y) for y in list(x)[::-1]) for x in list(samples_ndarray)]
    return samples


class Parities:
    """A class representing counts of parities for Pauli terms.

    Args:
        values (np.array): Number of observations of parities. See Attributes.
        correlations (list): Number of observations of pairwise products of terms.
            See Attributes.

    Attributes:
        values (np.array): an array of dimension N x 2 indicating how many times
            each Pauli term was observed with even and odd parity, where N is the
            number of Pauli terms. Here values[i][0] and values[i][1] correspond
            to the number of samples with even and odd parities for term P_i,
            respectively.
        correlations (list): a list of 3-dimensional numpy arrays indicating how
            many times each product of Pauli terms was observed with even and odd
            parity. Here correlations[i][j][k][0] and correlations[i][j][k][1]
            correspond to the number of samples with even and odd parities term P_j P_k
            in frame i, respectively.
    """

    def __init__(
        self, values: np.ndarray, correlations: Optional[List[np.ndarray]] = None
    ):
        self.values = values
        self.correlations = correlations

    def to_dict(self) -> dict:
        data: Dict[str, Any] = {"values": convert_array_to_dict(self.values)}
        if self.correlations:
            data["correlations"] = [
                convert_array_to_dict(arr) for arr in self.correlations
            ]
        return data

    @classmethod
    def from_dict(cls, data: dict):
        values = convert_dict_to_array(data["values"])
        if data.get("correlations"):
            correlations: Optional[List] = [
                convert_dict_to_array(arr) for arr in data["correlations"]
            ]
        else:
            correlations = None
        return cls(values, correlations)


def save_parities(parities: Parities, filename: AnyPath) -> None:
    """Save parities to a file.

    Args:
        parities (zquantum.core.measurement.Parities): the parities
        file (str or file-like object): the name of the file, or a file-like object
    """
    data = parities.to_dict()
    data["schema"] = SCHEMA_VERSION + "-parities"

    with open(filename, "w") as f:
        f.write(json.dumps(data, indent=2))


def load_parities(file: LoadSource) -> Parities:
    """Load parities from a file.

    Args:
        file (str or file-like object): the name of the file, or a file-like object.

    Returns:
        zquantum.core.measurement.Parities: the parities
    """

    if isinstance(file, str):
        with open(file, "r") as f:
            data = json.load(f)
    else:
        data = json.load(file)

    return Parities.from_dict(data)


def get_expectation_values_from_parities(parities: Parities) -> ExpectationValues:
    """Get the expectation values of a set of operators (with precisions) from a set of
    samples (with even/odd parities) for them.

    Args:
        parities: Contains the number of samples with even and odd parities for each
            operator.

    Returns:
        Expectation values of the operators and the associated precisions.
    """
    values = []
    estimator_covariances = []

    for i in range(len(parities.values)):
        N0 = parities.values[i][0]
        N1 = parities.values[i][1]
        N = N0 + N1
        if N == 0:
            raise ValueError("There must be at least one sample for each operator")

        p = N0 / N
        value = 2.0 * p - 1.0

        # If there are enough samples and the probability of getting a sample with even
        # parity is not close to 0 or 1, then we can use p=N0/N to approximate this
        # probability and plug it into the formula for the precision.
        if N >= 100 and p >= 0.1 and p <= 0.9:
            precision = 2.0 * np.sqrt(p * (1.0 - p)) / np.sqrt(N)
        else:
            # Otherwise, p=N0/N may be not a good approximation of this probability.
            # So we use an upper bound on the precision instead.
            precision = 1.0 / np.sqrt(N)

        values.append(value)
        estimator_covariances.append(np.array([[precision ** 2.0]]))

    return ExpectationValues(
        values=np.array(values), estimator_covariances=estimator_covariances
    )


def get_parities_from_measurements(
    measurements: List[Tuple[int]], ising_operator: IsingOperator
) -> Parities:
    """Get expectation values from bitstrings.

    Args:
        measurements (list): the measured bitstrings
        ising_operator (openfermion.ops.IsingOperator): the operator

    Returns:
        zquantum.core.measurement.Parities: the parities of each term in the operator
    """

    # check input format
    if not isinstance(ising_operator, IsingOperator):
        raise TypeError("Input operator not openfermion.IsingOperator")

    # Count number of occurrences of bitstrings
    bitstring_frequencies = Counter(measurements)

    # Count parity occurrences
    values = []
    for _, term in enumerate(ising_operator.terms):
        values.append([0, 0])
        marked_qubits = [op[0] for op in term]
        for bitstring, count in bitstring_frequencies.items():
            if check_parity(bitstring, marked_qubits):
                values[-1][0] += count
            else:
                values[-1][1] += count

    # Count parity occurrences for pairwise products of operators
    correlations = [np.zeros((len(ising_operator.terms), len(ising_operator.terms), 2))]
    for term1_index, term1 in enumerate(ising_operator.terms):
        for term2_index, term2 in enumerate(ising_operator.terms):
            marked_qubits_term1 = [op[0] for op in term1]
            marked_qubits_term2 = [op[0] for op in term2]
            for bitstring, count in bitstring_frequencies.items():
                parity1 = check_parity(bitstring, marked_qubits_term1)
                parity2 = check_parity(bitstring, marked_qubits_term2)
                if parity1 == parity2:
                    correlations[0][term1_index, term2_index][0] += count
                else:
                    correlations[0][term1_index, term2_index][1] += count

    return Parities(np.array(values), correlations)


def expectation_values_to_real(
    expectation_values: ExpectationValues,
) -> ExpectationValues:
    """Remove the imaginary parts of the expectation values

    Args:
        expectation_values (zquantum.core.measurement.ExpectationValues object)
    Returns:
        expectation_values (zquantum.core.measurement.ExpectationValues object)
    """
    values = []
    for i, value in enumerate(expectation_values.values):
        if isinstance(value, complex):
            value = value.real
        values.append(value)
    expectation_values.values = np.array(values)
    if expectation_values.correlations:
        for i, value in enumerate(expectation_values.correlations):
            if isinstance(value, complex):
                value = value.real
            expectation_values.correlations[i] = value
    return expectation_values


def convert_bitstring_to_int(bitstring: Sequence[int]) -> int:
    """Convert a bitstring to an integer.

    Args:
        bitstring (list): A list of integers.
    Returns:
        int: The value of the bitstring, where the first bit in the least
            significant (little endian).
    """
    return int("".join(str(bit) for bit in bitstring[::-1]), 2)


def check_parity(
    bitstring: Union[str, Sequence[int]], marked_qubits: Iterable[int]
) -> bool:
    """Determine if the marked qubits have even parity for the given bitstring.

    Args:
        bitstring: The bitstring, either as a tuple or little endian string.
        marked_qubits: The qubits whose parity is to be determined.

    Returns:
        True if an even number of the marked qubits are in the 1 state, False
            otherwise.
    """
    result = True
    for qubit_index in marked_qubits:
        if bitstring[qubit_index] == "1" or bitstring[qubit_index] == 1:
            result = not result
    return result


def get_expectation_value_from_frequencies(
    marked_qubits: Iterable[int], bitstring_frequencies: Dict[str, int]
) -> float:
    """Get the expectation value the product of Z operators on selected qubits
    from bitstring frequencies.

    Args:
        marked_qubits: The qubits that the Z operators act on.
        bitstring_frequences: The frequencies of the bitstrings.

    Returns:
        The expectation value of the product of Z operators on selected qubits.
    """

    expectation = 0.0
    num_measurements = sum(bitstring_frequencies.values())
    for bitstring, count in bitstring_frequencies.items():
        if check_parity(bitstring, marked_qubits):
            value = float(count) / num_measurements
        else:
            value = -float(count) / num_measurements
        expectation += value

    return expectation


class Measurements:
    """A class representing measurements from a quantum circuit. The bitstrings variable
    represents the internal data structure of the Measurements class. It is expressed as
    a list of tuples wherein each tuple is a measurement and the value of the tuple at a
    given index is the measured bit-value of the qubit (indexed from 0 -> N-1)"""

    def __init__(self, bitstrings: Optional[List[Tuple[int, ...]]] = None):
        if bitstrings is None:
            self.bitstrings = []
        else:
            self.bitstrings = bitstrings

    @classmethod
    def from_counts(cls, counts: Dict[str, int]):
        """Create an instance of the Measurements class from a dictionary

        Args:
            counts: mapping of bitstrings to integers representing the number of times
                the bitstring was measured
        """
        measurements = cls()
        measurements.add_counts(counts)
        return measurements

    @classmethod
    def get_measurements_representing_distribution(
        cls, bitstring_distribution: BitstringDistribution, number_of_samples: int
    ):
        """Create an instance of the Measurements class that exactly (or as closely as
        possible) resembles the input bitstring distribution.

        Args:
            bitstring_distribution: the bitstring distribution to be sampled
            number_of_samples: the number of measurements
        """
        distribution = copy.deepcopy(bitstring_distribution.distribution_dict)

        bitstring_samples = []
        # Rounding gives the closest integer to the observed frequency
        for state in distribution:
            bitstring = tuple([int(measurement_value) for measurement_value in state])

            bitstring_samples += [bitstring] * int(
                round(distribution[state] * number_of_samples)
            )

        # If frequencies are inconsistent with number of samples, we may need to
        # add or delete samples. The bitstrings to correct are chosen at random,
        # giving more weight to those with non-integer part closest to 0.5
        if len(bitstring_samples) != number_of_samples:
            leftover_distribution = BitstringDistribution(
                {
                    states: 0.5
                    - abs(0.5 - (distribution[states] * number_of_samples) % 1)
                    for states in distribution
                },
                True,
            )

            samples = sample_from_probability_distribution(
                leftover_distribution.distribution_dict,
                abs(number_of_samples - len(bitstring_samples)),
            )
            if number_of_samples - len(bitstring_samples) > 0:
                for sample in samples:
                    bitstring_samples += [
                        tuple([int(measurement_value) for measurement_value in sample])
                    ] * samples[sample]
            else:
                for sample in samples:
                    for _ in range(samples[sample]):
                        bitstring_samples.remove(
                            tuple(
                                [int(measurement_value) for measurement_value in sample]
                            )
                        )

        return cls(bitstring_samples)

    @classmethod
    def load_from_file(cls, file: TextIO):
        """Load a set of measurements from file

        Args:
            file (str or file-like object): the name of the file, or a file-like object
        """
        if isinstance(file, str):
            with open(file, "r") as f:
                data = json.load(f)
        else:
            data = json.load(file)

        bitstrings = []
        for bitstring in data["bitstrings"]:
            bitstrings.append(tuple(bitstring))

        return cls(bitstrings=bitstrings)

    def save(self, filename: AnyPath):
        """Serialize the Measurements object into a file in JSON format.

        Args:
            filename (string): filename to save the data to
        """
        data = {
            "schema": SCHEMA_VERSION + "-measurements",
            "counts": self.get_counts(),
            # This step is required if bistrings contain np.int8 instead of regular int.
            "bitstrings": [
                list(map(int, list(bitstring))) for bitstring in self.bitstrings
            ],
        }
        with open(filename, "w") as f:
            f.write(json.dumps(data, indent=2))

    def get_counts(self):
        """Get the measurements as a histogram

        Returns:
            A dictionary mapping bitstrings to integers representing the number of times
            the bitstring was measured
        """
        bitstrings = convert_tuples_to_bitstrings(self.bitstrings)
        return dict(Counter(bitstrings))

    def add_counts(self, counts: Dict[str, int]):
        """Add measurements from a histogram

        Args:
            counts: mapping of bitstrings to integers representing the number of times
                the bitstring was measured
                NOTE: bitstrings are also indexed from 0 -> N-1, where the "001"
                bitstring represents a measurement of qubit 2 in the 1 state
        """
        for bitstring in counts.keys():
            measurement = []
            for bitvalue in bitstring:
                measurement.append(int(bitvalue))

            self.bitstrings += [tuple(measurement)] * counts[bitstring]

    def get_distribution(self) -> BitstringDistribution:
        """Get the normalized probability distribution representing the measurements

        Returns:
            distribution: bitstring distribution based on the frequency of measurements
        """
        counts = self.get_counts()
        num_measurements = len(self.bitstrings)

        distribution = {}
        for bitstring in counts.keys():
            distribution[bitstring] = counts[bitstring] / num_measurements

        return BitstringDistribution(distribution)

    def get_expectation_values(
        self, ising_operator: IsingOperator, use_bessel_correction: bool = True
    ) -> ExpectationValues:
        """Get the expectation values of an operator from the measurements.

        Args:
            ising_operator: the operator
            use_bessel_correction: Whether to use Bessel's correction when
                when estimating the covariance of operators. Using the
                correction provides an unbiased estimate for covariances, but
                diverges when only one sample is taken.

        Returns:
            expectation values of each term in the operator
        """
        # We require operator to be IsingOperator because measurements are always
        # performed in the Z basis, so we need the operator to be Ising (containing only
        # Z terms). A general Qubit Operator could have X or Y terms which don’t get
        # directly measured.
        if not isinstance(ising_operator, IsingOperator):
            raise TypeError("Input operator is not openfermion.IsingOperator")

        # Count number of occurrences of bitstrings
        bitstring_frequencies = self.get_counts()
        num_measurements = len(self.bitstrings)

        # Perform weighted average
        expectation_values_list = [
            coefficient
            * get_expectation_value_from_frequencies(
                [cast(int, op[0]) for op in term], bitstring_frequencies
            )
            for term, coefficient in ising_operator.terms.items()
        ]
        expectation_values = np.array(expectation_values_list)

        correlations = np.zeros((len(ising_operator.terms),) * 2)
        for i, first_term in enumerate(ising_operator.terms):
            correlations[i, i] = ising_operator.terms[first_term] ** 2
            for j in range(i):
                second_term = list(ising_operator.terms.keys())[j]
                first_term_qubits = set(op[0] for op in first_term)
                second_term_qubits = set(op[0] for op in second_term)
                marked_qubits = first_term_qubits.symmetric_difference(
                    second_term_qubits
                )
                correlations[i, j] = (
                    ising_operator.terms[first_term]
                    * ising_operator.terms[second_term]
                    * get_expectation_value_from_frequencies(
                        marked_qubits, bitstring_frequencies
                    )
                )
                correlations[j, i] = correlations[i, j]

        denominator = (
            num_measurements - 1 if use_bessel_correction else num_measurements
        )

        estimator_covariances = (
            correlations
            - expectation_values[:, np.newaxis] * expectation_values[np.newaxis, :]
        ) / denominator

        return ExpectationValues(
            expectation_values, [correlations], [estimator_covariances]
        )


def concatenate_expectation_values(
    expectation_values_set: Iterable[ExpectationValues],
) -> ExpectationValues:
    """Concatenates a set of expectation values objects.

    Args:
        expectation_values_set: The expectation values objects to be concatenated.

    Returns:
        The combined expectation values.
    """

    combined_expectation_values = ExpectationValues(np.zeros(0))

    for expectation_values in expectation_values_set:
        combined_expectation_values.values = np.concatenate(
            (combined_expectation_values.values, expectation_values.values)
        )
        if expectation_values.correlations:
            if not combined_expectation_values.correlations:
                combined_expectation_values.correlations = []
            combined_expectation_values.correlations += expectation_values.correlations
        if expectation_values.estimator_covariances:
            if not combined_expectation_values.estimator_covariances:
                combined_expectation_values.estimator_covariances = []
            combined_expectation_values.estimator_covariances += (
                expectation_values.estimator_covariances
            )

    return combined_expectation_values
