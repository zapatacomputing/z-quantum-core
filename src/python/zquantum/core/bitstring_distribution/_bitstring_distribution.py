import json
import math
import sys
import warnings
from collections import Counter
from typing import Any, Callable, Dict, List

import numpy as np

from ..typing import AnyPath
from ..utils import SCHEMA_VERSION


class BitstringDistribution:
    """A probability distribution defined on discrete bitstrings. Normalization is
    performed by default, unless otherwise specified.

    Args:
        input_dict:  dictionary representing the probability distribution where
            the keys are bitstrings represented as strings and the values are
            non-negative floats.
        normalize: boolean variable specifying whether the input_dict gets
            normalized or not.
    Attributes:
        bitstring_distribution: dictionary representing the probability
            distribution where the keys are bitstrings represented as strings and the
            values are non-negative floats.
    """

    def __init__(self, input_dict: Dict, normalize: bool = True):
        if is_bitstring_distribution(
            input_dict
        ):  # accept the input dict only if it is a prob distribution
            if is_normalized(input_dict):
                self.distribution_dict = input_dict
            else:
                if normalize:
                    self.distribution_dict = normalize_bitstring_distribution(
                        input_dict
                    )
                else:
                    warnings.warn("BitstringDistribution object is not normalized.")
                    self.distribution_dict = input_dict
        else:
            raise RuntimeError(
                "Initialization of BitstringDistribution object FAILED: the input"
                " dictionary is not a bitstring probability distribution. Check keys"
                " (same-length binary strings) and values (non-negative floats)."
            )

    def __repr__(self) -> str:
        output = f"BitstringDistribution(input={self.distribution_dict})"
        return output

    def get_qubits_number(self) -> float:
        """Compute how many qubits a bitstring is composed of.

        Returns:
            float: number of qubits in a bitstring (i.e. bitstring length).
        """
        return len(
            list(self.distribution_dict.keys())[0]
        )  # already checked in __init__ that all keys have the same length


def is_non_negative(input_dict: Dict) -> bool:
    """Check if the input dictionary values are non negative.

    Args:
        input_dict (dict): dictionary.

    Returns:
        bool: boolean variable indicating whether dict values are non negative or not.
    """
    return all(value >= 0 for value in input_dict.values())


def is_key_length_fixed(input_dict: Dict) -> bool:
    """Check if the input dictionary keys are same-length.

    Args:
        input_dict (dict): dictionary.

    Returns:
        bool: boolean variable indicating whether dict keys are same-length or not.
    """
    key_length = len(list(input_dict.keys())[0])
    return all(len(key) == key_length for key in input_dict.keys())


def are_keys_binary_strings(input_dict: Dict) -> bool:
    """Check if the input dictionary keys are binary strings.

    Args:
        input_dict (dict): dictionary.

    Returns:
        bool: boolean variable indicating whether dict keys are binary strings or not.
    """
    return all(not any(char not in "10" for char in key) for key in input_dict.keys())


def is_bitstring_distribution(input_dict: Dict) -> bool:
    """Check if the input dictionary is a bitstring distribution, i.e.:
            - keys are same-lenght binary strings,
            - values are non negative.

    Args:
        input_dict: dictionary representing the probability distribution where the keys
            are bitstrings represented as strings and the values are floats.

    Returns:
        Boolean variable indicating whether the bitstring distribution is well
            defined or not.
    """
    return (
        (not input_dict == {})
        and is_non_negative(input_dict)
        and is_key_length_fixed(input_dict)
        and are_keys_binary_strings(input_dict)
    )


def is_normalized(input_dict: Dict) -> bool:
    """Check if a bitstring distribution is normalized.

    Args:
        bitstring_distribution: dictionary representing the probability distribution
            where the keys are bitstrings represented as strings and the values are
            floats.

    Returns:
        Boolean value indicating whether the bitstring distribution is normalized.
    """
    norm = sum(input_dict.values())
    return math.isclose(norm, 1)


def normalize_bitstring_distribution(bitstring_distribution: Dict) -> Dict:
    """Normalize a bitstring distribution.

    Args:
        bitstring_distribution: dictionary representing the probability
            distribution where the keys are bitstrings represented as strings and the
            values are floats.

    Returns:
        Dictionary representing the normalized probability distribution where the keys
            are bitstrings represented as strings and the values are floats.
    """
    norm = sum(bitstring_distribution.values())
    if norm == 0:
        raise ValueError(
            "Normalization of BitstringDistribution FAILED:"
            " input dict is empty (all zero values)."
        )
    elif 0 < norm < sys.float_info.min:
        raise ValueError(
            "Normalization of BitstringDistribution FAILED: too small values."
        )
    elif norm == 1:
        return bitstring_distribution
    else:
        for key in bitstring_distribution:
            bitstring_distribution[key] *= 1.0 / norm
        return bitstring_distribution


def save_bitstring_distribution(
    distribution: BitstringDistribution, filename: AnyPath
) -> None:
    """Save a bistring distribution to a file.

    Args:
        distribution (BitstringDistribution): the bistring distribution
        file (str or file-like object): the name of the file, or a file-like object
    """
    dictionary: Dict[str, Any] = {}
    dictionary["bitstring_distribution"] = distribution.distribution_dict
    dictionary["schema"] = SCHEMA_VERSION + "-bitstring-probability-distribution"

    with open(filename, "w") as f:
        f.write(json.dumps(dictionary, indent=2))


def save_bitstring_distributions(
    bitstring_distributions: List[BitstringDistribution], filename: str
) -> None:
    """Save a set of bitstring distributions to a file.

    Args:
       bitstring_distributions (list): a list of distributions to be saved
       file (str): the name of the file
    """
    dictionary: Dict[str, Any] = {}
    dictionary["schema"] = SCHEMA_VERSION + "-bitstring-probability-distribution-set"
    dictionary["bitstring_distribution"] = []

    for distribution in bitstring_distributions:
        dictionary["bitstring_distribution"].append(distribution.distribution_dict)

    with open(filename, "w") as f:
        f.write(json.dumps(dictionary, indent=2))


def load_bitstring_distribution(file: str) -> BitstringDistribution:
    """Load an bitstring_distribution from a json file using a schema.

    Arguments:
        file (str): the name of the file

    Returns:
        object: a python object loaded from the bitstring_distribution
    """
    if isinstance(file, str):
        with open(file, "r") as f:
            data = json.load(f)
    else:
        data = json.load(file)

    bitstring_distribution = BitstringDistribution(data["bitstring_distribution"])
    return bitstring_distribution


def load_bitstring_distributions(file: str) -> List[BitstringDistribution]:
    """Load a list of bitstring_distributions from a json file using a schema.

    Arguments:
        file: the name of the file.

    Returns:
        A list of bitstring distributions loaded from the bitstring_distribution
    """
    if isinstance(file, str):
        with open(file, "r") as f:
            data = json.load(f)
    else:
        data = json.load(file)

    bitstring_distribution_list = []
    for i in range(len(data["bitstring_distribution"])):
        bitstring_distribution_list.append(
            BitstringDistribution(data["bitstring_distribution"][i])
        )

    return bitstring_distribution_list


def create_bitstring_distribution_from_probability_distribution(
    prob_distribution: np.ndarray,
) -> BitstringDistribution:
    """Create a well defined bitstring distribution starting from a probability
    distribution.

    Args:
        probability distribution: The probabilities of the various states in the
            wavefunction.

    Returns:
        The BitstringDistribution object corresponding to the input measurements.
    """

    # Create dictionary of bitstring tuples as keys with probability as value
    prob_dict = {}
    for state in range(len(prob_distribution)):
        # Convert state to bitstring
        bitstring = format(state, "b")
        while len(bitstring) < np.log2(len(prob_distribution)):
            bitstring = "0" + bitstring
        # Reverse bitstring
        bitstring = bitstring[::-1]

        # Add to dict
        prob_dict[bitstring] = prob_distribution[state]

    return BitstringDistribution(prob_dict)


def evaluate_distribution_distance(
    target_distribution: BitstringDistribution,
    measured_distribution: BitstringDistribution,
    distance_measure_function: Callable,
    **kwargs,
) -> float:
    """Evaluate the distance between two bitstring distributions - the target
    distribution and the one predicted (measured) by your model - based on the given
    distance measure.

    Args:
         target_distribution: The target bitstring probability distribution
         measured_distribution: The measured bitstring probability distribution
         distance_measure_function: function used to calculate the distance measure
             Currently implemented: clipped negative log-likelihood, maximum mean
            discrepancy (MMD).

         Additional distance measure parameters can be passed as key word arguments.

    Returns:
         The value of the distance measure.
    """
    # Check inputs are BitstringDistribution objects
    if not isinstance(target_distribution, BitstringDistribution) or not isinstance(
        measured_distribution, BitstringDistribution
    ):
        raise TypeError(
            "Arguments of evaluate_cost_function must be of type BitstringDistribution."
        )

    # Check inputs are defined on consistent bitstring domains
    if (
        target_distribution.get_qubits_number()
        != measured_distribution.get_qubits_number()
    ):
        raise RuntimeError(
            "Bitstring Distribution Distance Evaluation FAILED: target "
            "and measured distributions are defined on bitstrings of different length."
        )

    # Check inputs are both normalized (or not normalized)
    if is_normalized(target_distribution.distribution_dict) != is_normalized(
        measured_distribution.distribution_dict
    ):
        raise RuntimeError(
            "Bitstring Distribution Distance Evaluation FAILED: one among target and"
            " measured distribution is normalized, whereas the other is not."
        )

    return distance_measure_function(
        target_distribution, measured_distribution, **kwargs
    )
