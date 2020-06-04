"""General-purpose utilities."""

import numpy as np
from scipy.linalg import expm
import random
import math
import operator
import sys
import json
import openfermion
from openfermion import hermitian_conjugated
from openfermion.ops import SymbolicOperator
from networkx.readwrite import json_graph
import lea
import collections
import scipy
from typing import List
import importlib

SCHEMA_VERSION = "zapata-v1"
RNDSEED = 12345


def convert_dict_to_array(dictionary: dict) -> np.ndarray:
    """Convert a dictionary to a numpy array.

    Args:
        dictionary (dict): the dict containing the data
    
    Returns:
        array (numpy.array): a numpy array
    """

    array = np.array(dictionary["real"])

    if dictionary.get("imag"):
        array = array + 1j * np.array(dictionary["imag"])

    return array


def convert_array_to_dict(array: np.ndarray) -> dict:
    """Convert a numpy array to a dictionary.

    Args:
        array (numpy.array): a numpy array
    
    Returns:
        dictionary (dict): the dict containing the data
    """

    dictionary = {}
    if np.iscomplexobj(array):
        dictionary["real"] = array.real.tolist()
        dictionary["imag"] = array.imag.tolist()
    else:
        dictionary["real"] = array.tolist()

    return dictionary


def dec2bin(number: int, length: int) -> List[int]:
    """Converts a decimal number into a binary representation
    of fixed number of bits.

    Args:
        number: (int) the input decimal number
        length: (int) number of bits in the output string

    Returns:
        A list of binary numbers
    """

    if pow(2, length) < number:
        sys.exit(
            "Insufficient number of bits for representing the number {}".format(number)
        )

    bit_str = bin(number)
    bit_str = bit_str[2 : len(bit_str)]  # chop off the first two chars
    bit_string = [int(x) for x in list(bit_str)]
    if len(bit_string) < length:
        len_zeros = length - len(bit_string)
        bit_string = [int(x) for x in list(np.zeros(len_zeros))] + bit_string

    return bit_string


def bin2dec(x: List[int]) -> int:
    """Converts a binary vector to an integer, with the 0-th
    element being the most significant digit.

    Args:
        x: (list) a binary vector

    Returns:
        An integer
    """

    dec = 0
    coeff = 1
    for i in range(len(x)):
        dec = dec + coeff * x[len(x) - 1 - i]
        coeff = coeff * 2
    return dec


"""
The functions PAULI_X, PAULI_Y, PAULI_Z and IDENTITY below are used for 
generating the generators of the Pauli group, which include Pauli X, Y, Z 
operators as well as identity operator
"""

pauli_x = np.array([[0.0, 1.0], [1.0, 0.0]])
pauli_y = np.array([[0.0, -1.0j], [1.0j, 0.0]])
pauli_z = np.array([[1.0, 0.0], [0.0, -1.0]])
identity = np.array([[1.0, 0.0], [0.0, 1.0]])


def is_identity(u, tol=1e-15):
    """Test if a matrix is identity.

    Args:
        u: np.ndarray
            Matrix to be checked.
        tol: float
            Threshold below which two matrix elements are considered equal.
    """

    dims = np.array(u).shape
    if dims[0] != dims[1]:
        raise Exception("Input matrix is not square.")

    return np.allclose(u, np.eye(u.shape[0]), atol=tol)


def is_unitary(u, tol=1e-15):
    """Test if a matrix is unitary.

    Args:
        u: array
            Matrix to be checked.
        tol: float
            Threshold below which two matrix elements are considered equal.
    """

    dims = np.array(u).shape
    if dims[0] != dims[1]:
        raise Exception("Input matrix is not square.")

    test_matrix = np.dot(hermitian_conjugated(np.array(u)), u)
    return is_identity(test_matrix, tol)


def compare_unitary(u1: np.ndarray, u2: np.ndarray, tol: float = 1e-15) -> bool:
    """Compares two unitary operators to see if they are equal to within a phase.

    Args:
        u1 (numpy.ndarray): First unitary operator.
        u2 (numpy.ndarray): Second unitary operator.
        tol (float): Threshold below which two matrix elements are considered equal.
    
    Returns:
        bool: True if the unitaries are equal to within the tolerance, ignoring
            differences in global phase.
    """

    if is_unitary(u1, tol) == False:
        raise Exception("The first input matrix is not unitary.")
    if is_unitary(u2, tol) == False:
        raise Exception("The second input matrix is not unitary.")

    test_matrix = np.dot(u1.conj().T, u2)
    phase = test_matrix.item((0, 0)) ** -1
    return is_identity(phase * test_matrix, tol)


def sample_from_probability_distribution(
    probability_distribution: dict, n_samples: int
) -> collections.Counter:
    """
    Samples events from a discrete probability distribution

    Args:
        probabilty_distribution: The discrete probability distribution to be used
        for sampling. This should be a dictionary
        
        n_samples (int): The number of samples desired

    Returns:
        A dictionary of the outcomes sampled. The key values are the things be sampled
        and values are how many times those things appeared in the sampling
    """
    if isinstance(probability_distribution, dict):
        prob_pmf = lea.pmf(probability_distribution)
        sampled_dict = collections.Counter(prob_pmf.random(n_samples))
        return sampled_dict
    else:
        raise RuntimeError(
            "Probability distribution should be a dictionary with key value \
        being the thing being sampled and the value being probability of getting \
        sampled "
        )


def convert_bitstrings_to_tuples(bitstrings):
    """Given the measured bitstrings, convert each bitstring to tuple format

    Args:
        bitstrings (list of strings): the measured bitstrings
    Returns:
        A list of tuples
    """
    # Convert from bitstrings to tuple format
    measurements = []
    for bitstring in bitstrings:

        measurement = ()
        for char in bitstring:
            measurement = measurement + (int(char),)

        measurements.append(measurement)
    return measurements


def convert_tuples_to_bitstrings(tuples):
    """Given a set of measurement tuples, convert each to bitstring format

    Args:
        tuples (list of tuples): the measurement tuples
    Returns:
        A list of bitstrings
    """
    # Convert from tuples to bitstrings
    bitstrings = []
    for tuple_item in tuples:

        bitstring = ""
        for bit in tuple_item:
            bitstring = bitstring + str(bit)

        bitstrings.append(bitstring)
    return bitstrings


class ValueEstimate:
    """A class representing a numerical value and its precision corresponding
        to an observable or an objective function

    Args:
        value (np.float): the numerical value
        precision (np.float): its precision

    Attributes:
        value (np.float): the numerical value
        precision (np.float): its precision
    """

    def __init__(self, value, precision=None):
        self.value = value
        self.precision = precision

    def to_dict(self):
        """Convert to a dictionary"""

        data = {"schema": SCHEMA_VERSION + "-value_estimate"}
        if type(self.value).__module__ == np.__name__:
            data["value"] = self.value.item()
        else:
            data["value"] = self.value

        if type(self.precision).__module__ == np.__name__:
            data["precision"] = self.precision.item()
        else:
            data["precision"] = self.precision

        return data

    @classmethod
    def from_dict(cls, dictionary):
        """Create an ExpectationValues object from a dictionary."""

        value = dictionary["value"]
        if "precision" in dictionary:
            precision = dictionary["precision"]
            return cls(value, precision)
        else:
            return cls(value)


def load_value_estimate(file):
    """Loads value estimate from a faile.

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

    return ValueEstimate.from_dict(data)


def save_value_estimate(value_estimate, filename):
    """Saves value estimate to a file.

    Args:
        value_estimate (core.utils.ValueEstimate): the value estimate
        file (str or file-like object): the name of the file, or a file-like object
    """
    dictionary = value_estimate.to_dict()
    dictionary["schema"] = SCHEMA_VERSION + "-value_estimate"

    with open(filename, "w") as f:
        f.write(json.dumps(dictionary, indent=2))


def load_list(file):
    """Load an array from a file.

    Args:
        file (str or file-like object): the name of the file, or a file-like object.
    
    Returns:
        array (list): the list
    """

    if isinstance(file, str):
        with open(file, "r") as f:
            data = json.load(f)
    else:
        data = json.load(file)

    return data["list"]


def save_list(array, filename):
    """Save expectation values to a file.

    Args:
        array (list): the list to be saved
        file (str or file-like object): the name of the file, or a file-like object
    """
    dictionary = {}
    dictionary["schema"] = SCHEMA_VERSION + "-list"
    dictionary["list"] = array

    with open(filename, "w") as f:
        f.write(json.dumps(dictionary, indent=2))


def create_object(specs, **kwargs):
    """
    Creates an object based on given specs.
    Specs include information about module and function necessary to create the object, 
    as well as any additional input parameters for it.

    Args:
        specs (dict): dictionary containing the following keys:
            module_name: specifies from which module an object comes.
            function_name: specifies the name of the function used to create object.
    
    Returns:
        object: object of any type
    """
    module_name = specs.pop("module_name")
    module = importlib.import_module(module_name)
    creator_name = specs.pop("function_name")
    creator = getattr(module, creator_name)
    created_object = creator(**specs, **kwargs)
    return created_object
