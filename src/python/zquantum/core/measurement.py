from __future__ import annotations
import json
from pyquil.wavefunction import Wavefunction
from grove.pyvqe.vqe import parity_even_p
import numpy as np
from openfermion.ops import IsingOperator
from .utils import (SCHEMA_VERSION, convert_array_to_dict, convert_dict_to_array,
    sample_from_probability_distribution, convert_bitstrings_to_tuples, convert_tuples_to_bitstrings)
from typing import Optional, List, Tuple, TextIO, Iterable
from collections import Counter
from .bitstring_distribution import BitstringDistribution

def save_expectation_values(expectation_values: np.ndarray, filename: str) -> None:
    """Save expectation values to a file.

    Args:
        array (numpy.array): the array
        file (str or file-like object): the name of the file, or a file-like object
    """
    dictionary = expectation_values.to_dict()
    dictionary['schema'] = SCHEMA_VERSION + '-expectation_values'

    with open(filename, 'w') as f:
        f.write(json.dumps(dictionary, indent=2))


def load_expectation_values(file: TextIO) -> ExpectationValues:
    """Load an array from a file.

    Args:
        file (str or file-like object): the name of the file, or a file-like object.

    Returns:
        array (numpy.array): the array
    """

    if isinstance(file, str):
        with open(file, 'r') as f:
            data = json.load(f)
    else:
        data = json.load(file)

    return ExpectationValues.from_dict(data)


def load_wavefunction(file: TextIO) -> Wavefunction:
    """Load a qubit wavefunction from a file.

    Args:
        file (str or file-like object): the name of the file, or a file-like object.

    Returns:
        wavefunction (pyquil.wavefunction.Wavefunction): the wavefunction object
    """

    if isinstance(file, str):
        with open(file, 'r') as f:
            data = json.load(f)
    else:
        data = json.load(file)

    wavefunction = Wavefunction(convert_dict_to_array(data['amplitudes']))
    return wavefunction


def save_wavefunction(wavefunction: Wavefunction, filename: str) -> None:
    """Save a wavefunction object to a file.

    Args:
        wavefunction (pyquil.wavefunction.Wavefunction): the wavefunction object
        filename (str): the name of the file
    """

    data = {'schema': SCHEMA_VERSION + '-wavefunction'}
    data['amplitudes'] = convert_array_to_dict(wavefunction.amplitudes)
    with open(filename, 'w') as f:
        f.write(json.dumps(data, indent=2))


class ExpectationValues:
    """A class representing expectation values of operators.

    Args:
        values (np.array): The expectation values of a set of operators.
        pairwise_products (list): The expectation values of pairwise products. Is a list of numpy.array objects.

    Attributes:
        values (np.array): The expectation values of a set of operators.
        correlations (list): The expectation values of pairwise products. Is a list of numpy.array objects, with each
            array corresponding to a frame. Is None if no correlations are available.
        covariances (list): The variances and covariances of different frames. Is a list of numpy.array
            objects, with each array corresponding to a frame. Is None if no covariances are available.
    """

    def __init__(self, values: np.ndarray, 
                correlations: Optional[List[np.ndarray]]=None, 
                covariances: Optional[List[np.ndarray]]=None
                ):
        self.values = values
        self.correlations = correlations
        self.covariances = covariances

    def to_dict(self) -> dict:
        """Convert to a dictionary"""

        data = {'schema' : SCHEMA_VERSION + '-expectation_values',
                'frames' : []}

        data['expectation_values'] = convert_array_to_dict(self.values)

        if self.correlations:
            data['correlations'] = []
            for correlation_matrix in self.correlations:
                data['correlations'].append(convert_array_to_dict(correlation_matrix))

        if self.covariances:
            data['covariances'] = []
            for covariance_matrix in self.covariances:
                data['covariances'].append(convert_array_to_dict(covariance_matrix))

        return data

    @classmethod
    def from_dict(cls, dictionary:dict) -> ExpectationValues:
        """Create an ExpectationValues object from a dictionary."""

        expectation_values = convert_dict_to_array(dictionary['expectation_values'])
        correlations = None
        if dictionary.get('correlations'):
            correlations = []
            for correlation_matrx in dictionary.get('correlations'):
                correlations.append(convert_dict_to_array(correlation_matrx))

        covariances = None
        if dictionary.get('covariances'):
            covariances = []
            for covariance_matrix in dictionary.get('covariances'):
                covariances.append(convert_dict_to_array(covariance_matrix))

        return cls(expectation_values, correlations, covariances)


def sample_from_wavefunction(wavefunction: Wavefunction, n_samples: int) -> List[Tuple[int]]:
    '''Sample bitstrings from a wavefunction
    Args:
        wavefunction (pyquil.wavefunction.Wavefunction): the wavefunction to
            sample from
        n_samples (int): the number of samples
    Returns:
        A list of bitstrings in tuple format
    '''
    # Get probabilities from pyquil
    probabilities = wavefunction.probabilities()

    # Create dictionary of bitstring tuples as keys with probability as value
    prob_dict = {}
    for state in range(len(probabilities)):
        # Convert state to bitstring
        bitstring = format(state, 'b')
        while (len(bitstring) < len(wavefunction)):
            bitstring = '0' + bitstring
        # Reverse bitstring
        bitstring = bitstring[::-1]
        # Convert bitstring to tuple
        bitstring_tuple = convert_bitstrings_to_tuples([bitstring])[0]
        # Add to dict
        prob_dict[bitstring_tuple] = probabilities[state]

    # Sample from dict
    samples_dict = sample_from_probability_distribution(prob_dict, n_samples)

    # Convert returned dict to tuples
    samples = []
    for key in samples_dict.keys():
        for _ in range(samples_dict[key]):
            samples.append(key)
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
            many times each product of Pauli terms was observed with even and odd parity.
            Here correlations[i][j][k][0] and correlations[i][j][k][1] correspond to the number
            of samples with even and odd parities term P_j P_k in frame i, respectively.
    """

    def __init__(self, values: np.ndarray, correlations: Optional[np.ndarray]=None):
        self.values = values
        self.correlations = correlations

    def to_dict(self) -> dict:
        data = {'values': convert_array_to_dict(self.values)}
        if self.correlations:
            data['correlations'] = [convert_array_to_dict(arr) for arr in self.correlations]
        return data

    @classmethod
    def from_dict(cls, data: dict):
        values = convert_dict_to_array(data['values'])
        if data.get('correlations'):
            correlations = [convert_dict_to_array(arr) for arr in data['correlations']]
        else:
            correlations = None
        return cls(values, correlations)


def save_parities(parities: Parities, filename: str) -> None:
    """Save parities to a file.

    Args:
        parities (zquantum.core.measurement.Parities): the parities
        file (str or file-like object): the name of the file, or a file-like object
    """
    data = parities.to_dict()
    data['schema'] = SCHEMA_VERSION + '-parities'

    with open(filename, 'w') as f:
        f.write(json.dumps(data, indent=2))


def load_parities(file: TextIO) -> Parities:
    """Load parities from a file.

    Args:
        file (str or file-like object): the name of the file, or a file-like object.

    Returns:
        zquantum.core.measurement.Parities: the parities
    """

    if isinstance(file, str):
        with open(file, 'r') as f:
            data = json.load(f)
    else:
        data = json.load(file)

    return Parities.from_dict(data)

def get_expectation_values_from_measurements(measurements: Measurements, 
                                            ising_operator: IsingOperator) -> ExpectationValues:
    """Get expectation values from bitstrings.

    Args:
        measurements (core.measurement.Measurements): the measured bitstrings
        ising_operator (openfermion.ops.IsingOperator): the operator

    Returns:
        zquantum.core.measurement.ExpectationValues: the expectation values of each term in the operator
    """
    # We require operator to be IsingOperator because measurements are always performed in the Z basis, 
    # so we need the operator to be Ising (containing only Z terms). 
    # A general Qubit Operator could have X or Y terms which don’t get directly measured.
    if isinstance(ising_operator, IsingOperator) == False:
        raise Exception("Input operator is not openfermion.IsingOperator")

    # Count number of occurrences of bitstrings
    bitstring_frequencies = measurements.get_counts()

    # Perform weighted average
    expectation_values = []
    for term, coefficient in ising_operator.terms.items():
        expectation = 0
        marked_qubits = [op[0] for op in term] 
        for bitstring, count in bitstring_frequencies.items():
            bitstring_int = convert_bitstring_to_int(bitstring)
            if parity_even_p(bitstring_int, marked_qubits):
                value = float(count)/len(measurements.bitstrings)
            else:
                value = -float(count)/len(measurements.bitstrings)
            expectation += np.real(coefficient) * value
        expectation_values.append(np.real(expectation))
    return ExpectationValues(np.array(expectation_values))

def get_expectation_values_from_parities(parities: Parities) -> ExpectationValues:
    """Get the expectation values of a set of operators (with precisions) from a set of samples (with even/odd parities) for them.

    Args:
        parities (zquantum.core.measurement.Parities): Contains the number of samples with even and odd parities for each operator.

    Returns:
        A zquantum.core.measurement.ExpectationValues object: Contains the expectation values of the operators and the associated precisions.
    """
    values = []
    covariances = []

    for i in range(len(parities.values)):
        N0 = parities.values[i][0]
        N1 = parities.values[i][1]
        N = N0 + N1
        if N == 0:
            raise ValueError('There must be at least one sample for each operator')

        p = N0 / N
        value = 2.0 * p - 1.0

        # If there are enough samples and the probability of getting a sample with even parity is not close to 0 or 1,
        # then we can use p=N0/N to approximate this probability and plug it into the formula for the precision.
        if N >= 100 and p >= 0.1 and p <= 0.9:
            precision = 2.0 * np.sqrt(p * (1.0 - p)) / np.sqrt(N)
        else:
            # Otherwise, p=N0/N may be not a good approximation of this probability.
            # So we use an upper bound on the precision instead.
            precision = 1.0 / np.sqrt(N)

        values.append(value)
        covariances.append(np.array([[precision ** 2.0]]))

    return ExpectationValues(values=np.array(values), covariances=covariances)

def get_parities_from_measurements(measurements: List[Tuple[int]], 
                                    ising_operator:IsingOperator) -> Parities:
    """Get expectation values from bitstrings.

    Args:
        measurements (list): the measured bitstrings
        ising_operator (openfermion.ops.IsingOperator): the operator

    Returns:
        zquantum.core.measurement.Parities: the parities of each term in the operator
    """

    # check input format
    if isinstance(ising_operator, IsingOperator) == False:
        raise Exception("Input operator not openfermion.IsingOperator")

    # Count number of occurrences of bitstrings
    bitstring_frequencies = Counter(measurements)

    # Count parity occurences
    values = []
    for _, term in enumerate(ising_operator.terms):
        values.append([0, 0])
        marked_qubits = [op[0] for op in term]
        for bitstring, count in bitstring_frequencies.items():
            bitstring_int = convert_bitstring_to_int(bitstring)
            if parity_even_p(bitstring_int, marked_qubits):
                values[-1][0] += count
            else:
                values[-1][1] += count

    # Count parity occurences for pairwise products of operators
    correlations = [np.zeros((len(ising_operator.terms), len(ising_operator.terms), 2))]
    for term1_index, term1 in enumerate(ising_operator.terms):
        for term2_index, term2 in enumerate(ising_operator.terms):
            marked_qubits_term1 = [op[0] for op in term1]
            marked_qubits_term2 = [op[0] for op in term2]
            for bitstring, count in bitstring_frequencies.items():
                bitstring_int = convert_bitstring_to_int(bitstring)
                parity1 = parity_even_p(bitstring_int, marked_qubits_term1)
                parity2 = parity_even_p(bitstring_int, marked_qubits_term2)
                if  parity1 == parity2:
                    correlations[0][term1_index, term2_index][0] += count
                else:
                    correlations[0][term1_index, term2_index][1] += count

    return Parities(np.array(values), correlations)

def expectation_values_to_real(expectation_values: ExpectationValues) -> ExpectationValues:
    """Remove the imaginary parts of the expectation values

    Args:
        expectation_values (zquantum.core.measurement.ExpectationValues object)
    Returns:
        expectation_values (zquantum.core.measurement.ExpectationValues object)
    """
    expectation_values.values = expectation_values.values.real
    if(expectation_values.correlations):
        for i, value in enumerate(expectation_values.correlations):
            if(isinstance(value, complex)):
                value = value.real
            expectation_values.correlations[i] = value
    return expectation_values

def convert_bitstring_to_int(bitstring: Iterable[int]) -> int:
    """Convert a bitstring to an integer.

    Args:
        bitstring (list): A list of integers.
    Returns:
        int: The value of the bitstring, where the first bit in the least
            significant (little endian).
    """
    return int("".join(str(bit) for bit in bitstring[::-1]), 2)

class Measurements:
    """ A class representing measurements from a quantum circuit. The bitstrings variable represents the internal
    data structure of the Measurements class. It is expressed as a list of tuples wherein each tuple is a measurement
    and the value of the tuple at a given index is the measured bit-value of the qubit (indexed from N-1 -> 0) """

    def __init__(self, bitstrings: Optional[List[Tuple[int]]] = None):
        if bitstrings is None:
            self.bitstrings = []
        else:
            self.bitstrings = bitstrings

    @classmethod
    def from_counts(cls, counts: Dict):
        """ Create an instance of the Measurements class from a dictionary
        
        Args:
            counts (dict): mapping of bitstrings to integers representing the number of times the bitstring was measured
        """
        measurements = cls()
        measurements.add_counts(counts)
        return measurements

    @classmethod
    def load_from_file(cls, file: TextIO):
        """ Load a set of measurements from file 
        
        Args:
            file (str or file-like object): the name of the file, or a file-like object
        """        
        if isinstance(file, str):
            with open(file, 'r') as f:
                data = json.load(f)
        else:
            data = json.load(file)

        bitstrings = []
        for bitstring in data["bitstrings"]:
            bitstrings.append(tuple(bitstring))

        return cls(bitstrings=bitstrings)

    def save(self, filename: String):
        """ Serialize the Measurements object into a file in JSON format.
        
        Args:
            filename (string): filename to save the data to 
        """
        data = { "schema":     SCHEMA_VERSION+"-measurements",
                 "counts":     self.get_counts(),
                 "bitstrings": self.bitstrings}
        with open(filename, "w") as f:
            f.write(json.dumps(data, indent=2))


    def get_counts(self):
        """ Get the measurements as a histogram

        Returns:
            A dictionary mapping bitstrings to integers representing the number of times the bitstring was measured
        """
        bitstrings = convert_tuples_to_bitstrings(self.bitstrings)
        return dict(Counter(bitstrings))


    def add_counts(self, counts: Dict):
        """ Add measurements from a histogram

        Args:
            counts (dict): mapping of bitstrings to integers representing the number of times the bitstring was measured
        """
        for bitstring in counts.keys():
            measurement = []
            for bitvalue in bitstring:
                measurement.append(int(bitvalue))

            self.bitstrings += [tuple(measurement)] * counts[bitstring]


    def get_distribution(self):
        """ Get the normalized probability distribution representing the measurements

        Returns:
            distribution (BitstringDistribution): bitstring distribution based on the frequency of measurements 
        """
        counts = self.get_counts()
        num_measurements = len(self.bitstrings)

        distribution = {}
        for bitstring in counts.keys():
            distribution[bitstring] = counts[bitstring]/num_measurements

        return BitstringDistribution(distribution)
