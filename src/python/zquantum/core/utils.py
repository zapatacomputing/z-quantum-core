"""General-purpose utilities."""
import warnings

import numpy as np
from scipy.linalg import expm
import random
import math
import operator
import sys
import json
import openfermion
import sympy
from openfermion import hermitian_conjugated
from openfermion import InteractionRDM
from openfermion.ops import SymbolicOperator
from networkx.readwrite import json_graph
import lea
import collections
import scipy
from typing import List, Tuple, Optional, Iterable, Union, Dict
import importlib
import copy
from .typing import LoadSource, AnyPath, Specs

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


def is_identity(u: np.ndarray, tol=1e-15) -> bool:
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


def is_unitary(u: np.ndarray, tol=1e-15) -> bool:
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


def convert_bitstrings_to_tuples(bitstrings: List[str]) -> List[Tuple[int]]:
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


def convert_tuples_to_bitstrings(tuples: List[Tuple[int]]) -> List[str]:
    """Given a set of measurement tuples, convert each to a little endian
    string.

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


class ValueEstimate(float):
    """A class representing a numerical value and its precision corresponding
        to an observable or an objective function

    Args:
        value (np.float): the numerical value or a value that can be converted to float
        precision (np.float): its precision

    Attributes:
        value (np.float): the numerical value
        precision (np.float): its precision
    """

    def __init__(self, value, precision: Optional[float] = None):
        super().__init__()
        self.precision = precision

    def __new__(cls, value, precision=None):
        return super().__new__(cls, value)

    @property
    def value(self):
        warnings.warn(
            "The value attribute is deprecated. Use ValueEstimate object directly instead.",
            DeprecationWarning,
        )
        return float(self)

    def __eq__(self, other):
        super_eq = super().__eq__(other)
        if super_eq is NotImplemented:
            return super_eq
        return super_eq and self.precision == getattr(other, "precision", None)

    def __ne__(self, other):
        return not self == other

    def __str__(self):
        value_str = super().__str__()
        if self.precision is not None:
            return f"{value_str} ± {self.precision}"
        else:
            return f"{value_str}"

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


def load_value_estimate(file: LoadSource) -> ValueEstimate:
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


def save_value_estimate(value_estimate: ValueEstimate, filename: AnyPath):
    """Saves value estimate to a file.

    Args:
        value_estimate (core.utils.ValueEstimate): the value estimate
        file (str or file-like object): the name of the file, or a file-like object
    """
    dictionary = value_estimate.to_dict()
    dictionary["schema"] = SCHEMA_VERSION + "-value_estimate"

    with open(filename, "w") as f:
        f.write(json.dumps(dictionary, indent=2))


def load_list(file: LoadSource) -> List:
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


def save_list(array: List, filename: AnyPath, artifact_name: str = ""):
    """Save expectation values to a file.

    Args:
        array (list): the list to be saved
        file (str or file-like object): the name of the file, or a file-like object
        artifact_name (str): optional argument to specify the schema name
    """
    dictionary = {}
    dictionary["schema"] = SCHEMA_VERSION + "-" + artifact_name + "-list"
    dictionary["list"] = array

    with open(filename, "w") as f:
        f.write(json.dumps(dictionary, indent=2))


def save_generic_dict(dictionary: Dict, filename: AnyPath):
    """Save dictionary as json

    Args:
        dictionary (dict): the dict containing the data
    """
    dictionary_stored = {"schema": SCHEMA_VERSION + "-dict"}
    dictionary_stored.update(dictionary)

    with open(filename, "w") as f:
        f.write(json.dumps(dictionary_stored, indent=2))


def get_func_from_specs(specs: Dict):
    """
    Return function based on given specs.
    Args:
        specs (dict): dictionary containing the following keys:
            module_name: specifies from which module an function comes.
            function_name: specifies the name of the function.

    Returns:
        callable: function defined by specs

    """
    module_name = specs.pop("module_name")
    module = importlib.import_module(module_name)
    function_name = specs.pop("function_name")
    return getattr(module, function_name)


def create_object(specs: Dict, **kwargs):
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
    specs = copy.copy(specs)
    module_name = specs.pop("module_name")
    module = importlib.import_module(module_name)
    creator_name = specs.pop("function_name")
    creator = getattr(module, creator_name)
    created_object = creator(**specs, **kwargs)
    return created_object


def load_noise_model(file: LoadSource):
    """Load a noise model from file

    Args:
        file (str or file-like object): the name of the file, or a file-like object.

    Returns:
        noise model
    """

    if isinstance(file, str):
        with open(file, "r") as f:
            data = json.load(f)
    else:
        data = json.load(file)

    func = get_func_from_specs(data)
    return func(data["data"])


def save_noise_model(noise_model_data, module_name, function_name, filename):
    """Save a noise model to file

    Args:
        noise_model_data (dict): the serialized version of the noise model
        module_name (str): the module name with the function to load the noise model
        function_name (str): the function to load the noise model data into a noise model
        filename (str or file-like object): the name of the file, or a file-like object.

    Returns:
        noise model
    """

    data = {
        "module_name": module_name,
        "function_name": function_name,
        "data": noise_model_data,
    }

    with open(filename, "w") as f:
        f.write(json.dumps(data, indent=2))


def create_symbols_map(
    symbols: List[sympy.Symbol], params: np.ndarray
) -> Tuple[sympy.Symbol, float]:
    """
    Creates a map to be used for evaluating sympy expressions.

    Args:
        symbols: list of sympy Symbols to be evaluated
        params: numpy array containing numerical value for the symbols
    """
    if len(symbols) != len(params):
        raise (
            ValueError(
                "Length of symbols: {0} doesn't match length of params: {1}".format(
                    len(symbols), len(params)
                )
            )
        )
    return [(symbol, param) for symbol, param in zip(symbols, params.tolist())]


def save_timing(walltime: float, filename: AnyPath) -> None:
    """
    Saves timing information.

    Args:
        walltime: The execution time.
    """

    with open(filename, "w") as f:
        f.write(
            json.dumps({"schema": SCHEMA_VERSION + "-timing", "walltime": walltime})
        )


def save_nmeas_estimate(
    nmeas: float, nterms: int, filename: AnyPath, frame_meas: np.ndarray = None
) -> None:
    """Save an estimate of the number of measurements to a file

    Args:
        nmeas: total number of measurements for epsilon = 1.0
        nterms: number of terms (groups) in the objective function
        frame_meas: A list of the number of measurements per frame for epsilon = 1.0
    """

    data = {}
    data["schema"] = SCHEMA_VERSION + "-hamiltonian_analysis"
    data["K"] = nmeas
    data["nterms"] = nterms
    if frame_meas is not None:
        data["frame_meas"] = convert_array_to_dict(frame_meas)

    with open(filename, "w") as f:
        f.write(json.dumps(data, indent=2))


def load_nmeas_estimate(filename: AnyPath) -> Tuple[float, int, np.ndarray]:
    """Load an estimate of the number of measurements from a file.

    Args:
        filename: the name of the file

    Returns:
        nmeas: number of measurements for epsilon = 1.0
        nterms: number of terms in the hamiltonian
        frame_meas: frame measurements (number of measurements per group)
    """

    with open(filename, "r") as f:
        data = json.load(f)

    frame_meas = convert_dict_to_array(data["frame_meas"])
    K_coeff = data["K"]
    nterms = data["nterms"]

    return K_coeff, nterms, frame_meas


def scale_and_discretize(values: Iterable[float], total: int) -> List[int]:
    """Convert a list of floats to a list of integers such that the total equals
    a given value and the ratios of elements are approximately preserved.

    Args:
        values: The list of floats to be scaled and discretized.
        total: The desired total which the resulting values should sum to.

    Returns:
        A list of integers whose sum is equal to the given total, where the
            ratios of the list elements are approximately equal to the ratios
            of the input list elements.
    """

    scale_factor = total / sum(values)

    result = [np.floor(value * scale_factor) for value in values]
    remainders = [
        value * scale_factor - np.floor(value * scale_factor) for value in values
    ]
    indexes_sorted_by_remainder = np.argsort(remainders)[::-1]
    for index in range(int(round(total - sum(result)))):
        result[indexes_sorted_by_remainder[index]] += 1

    result = [int(value) for value in result]

    assert sum(result) == total, "The scaled list does not sum to the desired total."

    return result


def hf_rdm(n_alpha: int, n_beta: int, n_orbitals: int) -> InteractionRDM:
    """Construct the RDM corresponding to a Hartree-Fock state.

    Args:
        n_alpha (int): number of spin-up electrons
        n_beta (int): number of spin-down electrons
        n_orbitals (int): number of spatial orbitals (not spin orbitals)

    Returns:
        openfermion.ops.InteractionRDM: the reduced density matrix
    """
    # Determine occupancy of each spin orbital
    occ = np.zeros(2 * n_orbitals)
    occ[: (2 * n_alpha) : 2] = 1
    occ[1 : (2 * n_beta + 1) : 2] = 1

    one_body_tensor = np.diag(occ)

    two_body_tensor = np.zeros([2 * n_orbitals for i in range(4)])
    for i in range(2 * n_orbitals):
        for j in range(2 * n_orbitals):
            if i != j and occ[i] and occ[j]:
                two_body_tensor[i, j, j, i] = 1
                two_body_tensor[i, j, i, j] = -1

    return InteractionRDM(one_body_tensor, two_body_tensor)


def load_from_specs(specs: Specs):
    if isinstance(specs, str):
        specs = json.loads(specs)
    return create_object(specs)
