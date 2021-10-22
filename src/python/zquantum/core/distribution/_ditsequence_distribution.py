import json
import math
import sys
import warnings
from itertools import product
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np

from ..typing import AnyPath
from ..utils import SCHEMA_VERSION


class DitSequenceDistribution:
    """A probability distribution defined on discrete dit sequences. Normalization is
    performed by default, unless otherwise specified.

    Args:
        input_dict:  dictionary representing the probability distribution where
            the keys are dit sequences represented as tuples and the values are
            non-negative floats.
        normalize: boolean variable specifying whether the input_dict gets
            normalized or not.
    Attributes:
        ditstring_distribution: dictionary representing the probability
            distribution where the keys are ditstrings represented as tuples and the
            values are non-negative floats.
    """

    def __init__(
        self,
        input_dict: Union[
            Dict[Tuple[int, ...], float],
            Dict[str, float],
            Dict[Union[str, Tuple[int, ...]], float],
        ],
        normalize: bool = True,
    ):
        # First check if we are initializing from binary strings
        preprocessed_input_dict = preprocess_distibution_dict(input_dict)

        if is_ditsequence_distribution(
            preprocessed_input_dict
        ):  # accept the input dict only if it is a prob distribution
            if is_normalized(preprocessed_input_dict):
                self.distribution_dict = preprocessed_input_dict
            else:
                if normalize:
                    self.distribution_dict = normalize_ditstring_distribution(
                        preprocessed_input_dict
                    )
                else:
                    warnings.warn("DitSequenceDistribution object is not normalized.")
                    self.distribution_dict = preprocessed_input_dict
        else:
            raise RuntimeError(
                "Initialization of DitSequenceDistribution object FAILED: the input"
                " dictionary is not a dit sequence probability distribution. Check keys"
                " (same-length dit tuples) and values (non-negative floats)."
            )

    def __repr__(self) -> str:
        output = f"DitSequenceDistribution(input={self.distribution_dict})"
        return output

    def get_qubits_number(self) -> float:
        """Compute how many qubits a dit sequence is composed of.

        Returns:
            float: number of qubits in a dit sequence (i.e. dit sequence length).
        """
        return len(
            list(self.distribution_dict.keys())[0]
        )  # already checked in __init__ that all keys have the same length


def preprocess_distibution_dict(
    input_dict: Union[
        Dict[Tuple[int, ...], float],
        Dict[str, float],
        Dict[Union[str, Tuple[int, ...]], float],
    ]
) -> Dict[Tuple[int, ...], float]:
    res_dict = {}
    for key, value in input_dict.items():
        if isinstance(key, str):
            res_dict[tuple(map(int, key if "," not in key else key.split(",")))] = value
        elif isinstance(key, tuple):
            res_dict[key] = value
        else:
            raise RuntimeError(
                "Initialization of DitSequenceDistribution object FAILED:"
                "The non-tuple dictionary keys are not consistent. "
                "Check that the keys are either same-length tuples, binary strings "
                "or comma-separated non-negative integer strings."
            )

    return res_dict


def are_non_tuple_keys_valid_binary_strings(
    input_dict: Dict[Union[str, Tuple[int, ...]], float]
) -> bool:
    """Check if any of the input dictionary keys are valid binary strings.

    Args:
        input_dict (dict): dictionary.

    Returns:
        bool: boolean variable indicating whether non-tuple dict keys are valid.
    """
    non_tuple_keys = {key for key in input_dict.keys() if not isinstance(key, tuple)}

    # 1. Check that all non-tuples are strings
    # 2. Check the strings are binary
    return all(isinstance(key, str) for key in non_tuple_keys) and all(
        not any(char not in "10" for char in key) for key in non_tuple_keys
    )


def is_non_negative(input_dict: Dict[Tuple[int, ...], float]) -> bool:
    """Check if the input dictionary values are non negative.

    Args:
        input_dict (dict): dictionary.

    Returns:
        bool: boolean variable indicating whether dict values are non negative or not.
    """
    return all(value >= 0 for value in input_dict.values())


def is_key_length_fixed(input_dict: Dict[Tuple[int, ...], float]) -> bool:
    """Check if the input dictionary keys are same-length.

    Args:
        input_dict (dict): dictionary.

    Returns:
        bool: boolean variable indicating whether dict keys are same-length or not.
    """
    key_length = len(list(input_dict.keys())[0])
    return all(len(key) == key_length for key in input_dict.keys())


def are_keys_non_negative_integer_tuples(
    input_dict: Dict[Tuple[int, ...], float]
) -> bool:
    """Check if the input dictionary keys are tuples containing non-negative integers.

    Args:
        input_dict (dict): dictionary.

    Returns:
        bool: boolean variable indicating whether dict keys are binary strings or not.
    """
    return all(
        all(isinstance(sub, int) and sub >= 0 for sub in key)
        for key in input_dict.keys()
    )


def is_ditsequence_distribution(input_dict: Dict[Tuple[int, ...], float]) -> bool:
    """Check if the input dictionary is a dit sequence distribution, i.e.:
            - keys are same-length tuples of non-negative integers,
            - values are non negative.

    Args:
        input_dict: dictionary representing the probability distribution where the keys
            are dit sequences represented as tuples and the values are floats.

    Returns:
        Boolean variable indicating whether the dit sequence distribution is well
            defined or not.
    """
    return (
        (not input_dict == {})
        and is_non_negative(input_dict)
        and is_key_length_fixed(input_dict)
        and are_keys_non_negative_integer_tuples(input_dict)
    )


def is_normalized(input_dict: Dict[Tuple[int, ...], float]) -> bool:
    """Check if a dit sequence distribution is normalized.

    Args:
        input_dict: dictionary representing the probability distribution
            where the keys are dit sequences represented as strings and the values are
            floats.

    Returns:
        Boolean value indicating whether the dit sequence distribution is normalized.
    """
    norm = sum(input_dict.values())
    return math.isclose(norm, 1)


def normalize_ditstring_distribution(
    ditsequence_distribution: Dict[Tuple[int, ...], float]
) -> Dict:
    """Normalize a dit sequence distribution.

    Args:
        ditsequence_distribution: dictionary representing the probability
            distribution where the keys are dit sequences represented as strings and the
            values are floats.

    Returns:
        Dictionary representing the normalized probability distribution where the keys
            are dit sequences represented as strings and the values are floats.
    """
    norm = sum(ditsequence_distribution.values())
    if norm == 0:
        raise ValueError(
            "Normalization of DitSequenceDistribution FAILED:"
            " input dict is empty (all zero values)."
        )
    elif 0 < norm < sys.float_info.min:
        raise ValueError(
            "Normalization of DitSequenceDistribution FAILED: too small values."
        )
    elif norm == 1:
        return ditsequence_distribution
    else:
        for key in ditsequence_distribution:
            ditsequence_distribution[key] *= 1.0 / norm
        return ditsequence_distribution


def change_tuple_dict_keys_to_comma_separated_digitstrings(dict):
    return {
        ",".join(map(str, key)) if isinstance(key, tuple) else key: value
        for key, value in dict.items()
    }


def save_ditsequence_distribution(
    distribution: DitSequenceDistribution, filename: AnyPath
) -> None:
    """Save a bistring distribution to a file.

    Args:
        distribution (DitSequenceDistribution): the bistring distribution
        file (str or file-like object): the name of the file, or a file-like object
    """
    dictionary: Dict[str, Any] = {}

    # Need to convert tuples to strings for JSON encoding
    preprocessed_distribution_dict = (
        change_tuple_dict_keys_to_comma_separated_digitstrings(
            distribution.distribution_dict
        )
    )

    dictionary["ditsequence_distribution"] = preprocessed_distribution_dict
    dictionary["schema"] = SCHEMA_VERSION + "-ditsequence-probability-distribution"
    with open(filename, "w") as f:
        f.write(json.dumps(dictionary, indent=2))


def save_ditsequence_distributions(
    ditsequence_distributions: List[DitSequenceDistribution], filename: str
) -> None:
    """Save a set of dit sequence distributions to a file.

    Args:
       ditsequence_distributions (list): a list of distributions to be saved
       file (str): the name of the file
    """
    dictionary: Dict[str, Any] = {}
    dictionary["schema"] = SCHEMA_VERSION + "-ditsequence-probability-distribution-set"
    dictionary["ditsequence_distribution"] = []

    for distribution in ditsequence_distributions:
        dictionary["ditsequence_distribution"].append(
            change_tuple_dict_keys_to_comma_separated_digitstrings(
                distribution.distribution_dict
            )
        )

    with open(filename, "w") as f:
        f.write(json.dumps(dictionary, indent=2))


def load_ditsequence_distribution(file: str) -> DitSequenceDistribution:
    """Load a dit sequence from a json file using a schema.

    Arguments:
        file (str): the name of the file

    Returns:
        object: a python object loaded from the ditsequence_distribution
    """
    if isinstance(file, str):
        with open(file, "r") as f:
            data = json.load(f)
    else:
        data = json.load(file)

    try:
        distribution = DitSequenceDistribution(data["bitstring_distribution"])
    except KeyError:
        distribution = DitSequenceDistribution(data["ditsequence_distribution"])
    return distribution


def load_ditsequence_distributions(file: str) -> List[DitSequenceDistribution]:
    """Load a list of ditsequence_distributions from a json file using a schema.

    Arguments:
        file: the name of the file.

    Returns:
        A list of dit sequence distributions loaded from the ditsequence_distribution
    """
    if isinstance(file, str):
        with open(file, "r") as f:
            data = json.load(f)
    else:
        data = json.load(file)

    distribution_list = []
    for i in range(len(data["ditsequence_distribution"])):
        try:
            distribution_list.append(
                DitSequenceDistribution(data["bitstring_distribution"][i])
            )
        except KeyError:
            distribution_list.append(
                DitSequenceDistribution(data["ditsequence_distribution"][i])
            )

    return distribution_list


def create_bitstring_distribution_from_probability_distribution(
    prob_distribution: np.ndarray,
) -> DitSequenceDistribution:
    """Create a well defined bitstring distribution starting from a probability
    distribution.

    Args:
        probability distribution: The probabilities of the various states in the
            wavefunction.

    Returns:
        The DitSequenceDistribution object corresponding to the input measurements.
    """
    # Create dictionary of bitstring tuples as keys with probability as value
    keys = product([0, 1], repeat=int(np.log2(len(prob_distribution))))
    prob_dict: Dict[Union[str, Tuple[int, ...]], float] = {
        key: float(value) for key, value in zip(keys, prob_distribution)
    }

    return DitSequenceDistribution(prob_dict)


def evaluate_distribution_distance(
    target_distribution: DitSequenceDistribution,
    measured_distribution: DitSequenceDistribution,
    distance_measure_function: Callable,
    **kwargs,
) -> float:
    """Evaluate the distance between two dit sequence distributions - the target
    distribution and the one predicted (measured) by your model - based on the given
    distance measure.

    Args:
         target_distribution: The target dit sequence probability distribution
         measured_distribution: The measured dit sequence probability distribution
         distance_measure_function: function used to calculate the distance measure
             Currently implemented: clipped negative log-likelihood, maximum mean
            discrepancy (MMD).

         Additional distance measure parameters can be passed as key word arguments.

    Returns:
         The value of the distance measure.
    """
    # Check inputs are DitSequenceDistribution objects
    if not isinstance(target_distribution, DitSequenceDistribution) or not isinstance(
        measured_distribution, DitSequenceDistribution
    ):
        raise TypeError(
            "Arguments of evaluate_cost_function must"
            " be of type DitSequenceDistribution."
        )

    # Check inputs are defined on consistent dit sequence domains
    if (
        target_distribution.get_qubits_number()
        != measured_distribution.get_qubits_number()
    ):
        raise RuntimeError(
            "Dit Sequence Distribution Distance Evaluation FAILED: target "
            "and measured distributions are defined on "
            "dit sequences of different length."
        )

    # Check inputs are both normalized (or not normalized)
    if is_normalized(target_distribution.distribution_dict) != is_normalized(
        measured_distribution.distribution_dict
    ):
        raise RuntimeError(
            "Dit Sequence Distribution Distance Evaluation FAILED: one among target and"
            " measured distribution is normalized, whereas the other is not."
        )

    return distance_measure_function(
        target_distribution, measured_distribution, **kwargs
    )
