import json
import math
import sys
import warnings
from itertools import product
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np

from ..typing import AnyPath
from ..utils import SCHEMA_VERSION


class MeasurementOutcomeDistribution:
    """A probability distribution defined on discrete non-negative integer
    sequences. Normalization is performed by default, unless otherwise specified.

    Args:
        input_dict:  dictionary representing the probability distribution where
            the keys are non-negative integer sequences represented as tuples
            and the values are non-negative floats.
        normalize: boolean variable specifying whether the input_dict gets
            normalized or not.
    Attributes:
        distribution_dict: dictionary representing the probability
            distribution where the keys are non-negative integer sequences
            represented as tuples and the values are non-negative floats.
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

        if is_measurement_outcome_distribution(
            preprocessed_input_dict
        ):  # accept the input dict only if it is a prob distribution
            if is_normalized(preprocessed_input_dict):
                self.distribution_dict = preprocessed_input_dict
            else:
                if normalize:
                    self.distribution_dict = normalize_measurement_outcome_distribution(
                        preprocessed_input_dict
                    )
                else:
                    warnings.warn(
                        "MeasurementOutcomeDistribution object is not normalized."
                    )
                    self.distribution_dict = preprocessed_input_dict
        else:
            raise RuntimeError(
                "Initialization of MeasurementOutcomeDistribution object FAILED: "
                "the input dictionary is not a non-negative integer sequence "
                "probability distribution. Check keys (same-length non-negative integer"
                " tuples) and values (non-negative floats)."
            )

    def __repr__(self) -> str:
        output = f"MeasurementOutcomeDistribution(input={self.distribution_dict})"
        return output

    def get_number_of_subsystems(self) -> int:
        """Compute how many subsystems the measurement outcome is composed of.
        This corresponds to the number of qubits in a digital quantum computer.

        Returns:
            int: number of subsystems in the measurement outcome
                (i.e. non-negative integer sequence length).
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
                "Initialization of MeasurementOutcomeDistribution object FAILED:"
                "The non-tuple dictionary keys are not consistent. "
                "Check that the keys are either same-length tuples, binary strings "
                "or comma-separated non-negative integer strings."
            )

    return res_dict


def _is_non_negative(input_dict: Dict[Tuple[int, ...], float]) -> bool:
    """Check if the input dictionary values are non-negative.

    Args:
        input_dict (dict): dictionary.

    Returns:
        bool: boolean variable indicating whether dict values are non negative or not.
    """
    return all(value >= 0 for value in input_dict.values())


def _is_key_length_fixed(input_dict: Dict[Tuple[int, ...], float]) -> bool:
    """Check if the input dictionary keys are same-length.

    Args:
        input_dict (dict): dictionary.

    Returns:
        bool: boolean variable indicating whether dict keys are same-length or not.
    """
    key_length = len(list(input_dict.keys())[0])
    return all(len(key) == key_length for key in input_dict.keys())


def _are_keys_non_negative_integer_tuples(
    input_dict: Dict[Tuple[int, ...], float]
) -> bool:
    """Check if the input dictionary keys are tuples containing non-negative integers.

    Args:
        input_dict (dict): dictionary.

    Returns:
        bool: boolean variable indicating whether dict keys are binary strings or not.
    """
    return all(
        all(isinstance(sub, (int, np.integer)) and sub >= 0 for sub in key)
        for key in input_dict.keys()
    )


def is_measurement_outcome_distribution(
    input_dict: Dict[Tuple[int, ...], float]
) -> bool:
    """Check if the input dictionary is a non-negative integer sequence distribution, i.e.:
            - keys are same-length tuples of non-negative integers,
            - values are non negative.

    Args:
        input_dict: dictionary representing the probability distribution where
            the keys are non-negative integer sequences represented as tuples
            and the values are floats.

    Returns:
        Boolean variable indicating whether the non-negative integer sequence
            distribution is well defined or not.
    """
    return (
        (not input_dict == {})
        and _is_non_negative(input_dict)
        and _is_key_length_fixed(input_dict)
        and _are_keys_non_negative_integer_tuples(input_dict)
    )


def is_normalized(input_dict: Dict[Tuple[int, ...], float]) -> bool:
    """Check if a measurement outcome distribution is normalized.

    Args:
        input_dict: dictionary representing the probability distribution
            where the keys are non-negative integer sequences represented
            as tuples and the values are floats.

    Returns:
        Boolean value indicating whether the measurement outcome distribution
            is normalized.
    """
    norm = sum(input_dict.values())
    return math.isclose(norm, 1)


def normalize_measurement_outcome_distribution(
    measurement_outcome_distribution: Dict[Tuple[int, ...], float]
) -> Dict:
    """Normalize a measurement outcome distribution.

    Args:
        measurement_outcome_distribution: dictionary representing the probability
            distribution where the keys are non-negative integer sequences represented
            as tuples and the values are floats.

    Returns:
        Dictionary representing the normalized probability distribution where the keys
            are non-negative integer sequences represented as tuples and the values
            are floats.
    """
    norm = sum(measurement_outcome_distribution.values())
    if norm == 0:
        raise ValueError(
            "Normalization of MeasurementOutcomeDistribution FAILED:"
            " input dict is empty (all zero values)."
        )
    elif 0 < norm < sys.float_info.min:
        raise ValueError(
            "Normalization of MeasurementOutcomeDistribution FAILED: too small values."
        )
    elif norm == 1:
        return measurement_outcome_distribution
    else:
        for key in measurement_outcome_distribution:
            measurement_outcome_distribution[key] *= 1.0 / norm
        return measurement_outcome_distribution


def change_tuple_dict_keys_to_comma_separated_integers(dict):
    return {
        ",".join(map(str, key)) if isinstance(key, tuple) else key: value
        for key, value in dict.items()
    }


def save_measurement_outcome_distribution(
    distribution: MeasurementOutcomeDistribution, filename: AnyPath
) -> None:
    """Save a measurement outcome distribution to a file.

    Args:
        distribution (MeasurementOutcomeDistribution): the measurement outcome
        distribution file (str or file-like object): the name of the file,
            or a file-like object
    """
    dictionary: Dict[str, Any] = {}

    # Need to convert tuples to strings for JSON encoding
    preprocessed_distribution_dict = change_tuple_dict_keys_to_comma_separated_integers(
        distribution.distribution_dict
    )

    dictionary["measurement_outcome_distribution"] = preprocessed_distribution_dict
    dictionary["schema"] = (
        SCHEMA_VERSION + "-measurement-outcome-probability-distribution"
    )
    with open(filename, "w") as f:
        f.write(json.dumps(dictionary, indent=2))


def save_measurement_outcome_distributions(
    measurement_outcome_distribution: List[MeasurementOutcomeDistribution],
    filename: str,
) -> None:
    """Save a set of measurement outcome distributions to a file.

    Args:
       measurement_outcome_distribution (list): a list of distributions to be saved
       file (str): the name of the file
    """
    dictionary: Dict[str, Any] = {}
    dictionary["schema"] = (
        SCHEMA_VERSION + "-measurement-outcome-probability-distribution-set"
    )
    dictionary["measurement_outcome_distribution"] = []

    for distribution in measurement_outcome_distribution:
        dictionary["measurement_outcome_distribution"].append(
            change_tuple_dict_keys_to_comma_separated_integers(
                distribution.distribution_dict
            )
        )

    with open(filename, "w") as f:
        f.write(json.dumps(dictionary, indent=2))


def load_measurement_outcome_distribution(file: str) -> MeasurementOutcomeDistribution:
    """Load a measurement outcome distribution from a json file using a schema.

    Arguments:
        file (str): the name of the file

    Returns:
        object: a python object loaded from the measurement_outcome_distribution
    """
    if isinstance(file, str):
        with open(file, "r") as f:
            data = json.load(f)
    else:
        data = json.load(file)

    try:
        distribution = MeasurementOutcomeDistribution(data["bitstring_distribution"])
    except KeyError:
        distribution = MeasurementOutcomeDistribution(
            data["measurement_outcome_distribution"]
        )
    return distribution


def load_measurement_outcome_distributions(
    file: str,
) -> List[MeasurementOutcomeDistribution]:
    """Load a list of measurement_outcome_distribution from a json file using a schema.

    Arguments:
        file: the name of the file.

    Returns:
        A list of measurement outcome distributions loaded
         from the measurement_outcome_distribution
    """
    if isinstance(file, str):
        with open(file, "r") as f:
            data = json.load(f)
    else:
        data = json.load(file)

    distribution_list = []
    for i in range(len(data["measurement_outcome_distribution"])):
        try:
            distribution_list.append(
                MeasurementOutcomeDistribution(data["bitstring_distribution"][i])
            )
        except KeyError:
            distribution_list.append(
                MeasurementOutcomeDistribution(
                    data["measurement_outcome_distribution"][i]
                )
            )

    return distribution_list


def create_bitstring_distribution_from_probability_distribution(
    prob_distribution: np.ndarray,
) -> MeasurementOutcomeDistribution:
    """Create a well defined bitstring distribution starting from a probability
    distribution.

    Args:
        probability distribution: The probabilities of the various states in the
            wavefunction.

    Returns:
        The MeasurementOutcomeDistribution object corresponding to
            the input measurements.
    """
    # Create dictionary of bitstring tuples as keys with probability as value
    keys = product([0, 1], repeat=int(np.log2(len(prob_distribution))))
    prob_dict: Dict[Union[str, Tuple[int, ...]], float] = {
        key: float(value) for key, value in zip(keys, prob_distribution)
    }

    return MeasurementOutcomeDistribution(prob_dict)


def evaluate_distribution_distance(
    target_distribution: MeasurementOutcomeDistribution,
    measured_distribution: MeasurementOutcomeDistribution,
    distance_measure_function: Callable,
    **kwargs,
) -> float:
    """Evaluate the distance between two measurement outcome distributions - the target
    distribution and the one predicted (measured) by your model - based on the given
    distance measure.

    Args:
         target_distribution: The target measurement outcome probability distribution
         measured_distribution: The measured measurement outcome probability
          distribution
         distance_measure_function: function used to calculate the distance measure
             Currently implemented: clipped negative log-likelihood, maximum mean
            discrepancy (MMD).

         Additional distance measure parameters can be passed as key word arguments.

    Returns:
         The value of the distance measure.
    """
    # Check inputs are MeasurementOutcomeDistribution objects
    if not isinstance(
        target_distribution, MeasurementOutcomeDistribution
    ) or not isinstance(measured_distribution, MeasurementOutcomeDistribution):
        raise TypeError(
            "Arguments of evaluate_cost_function must"
            " be of type MeasurementOutcomeDistribution."
        )

    # Check inputs are defined on consistent measurement outcome domains
    if (
        target_distribution.get_number_of_subsystems()
        != measured_distribution.get_number_of_subsystems()
    ):
        raise RuntimeError(
            "Measurement Outcome Distribution Distance Evaluation FAILED: target "
            "and measured distributions are defined on tuples of different length."
        )

    # Check inputs are both normalized (or not normalized)
    if is_normalized(target_distribution.distribution_dict) != is_normalized(
        measured_distribution.distribution_dict
    ):
        raise RuntimeError(
            "Measurement Outcome Distribution Distance Evaluation FAILED: one among "
            "target and measured distribution is normalized, whereas the other is not."
        )

    return distance_measure_function(
        target_distribution, measured_distribution, **kwargs
    )
