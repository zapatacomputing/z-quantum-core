import math
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from zquantum.core.distribution import MeasurementOutcomeDistribution


def total_variation_distance(
    target_distribution: "MeasurementOutcomeDistribution",
    measured_distribution: "MeasurementOutcomeDistribution"
) -> float:
    """Compute the total variation distance between two distributions,
    potentially a target distribution and a measured distribution.

    Args:
        target_distribution: The target probability distribution.
        measured_distribution: The measured probability distribution.

    Returns:
        The value of the the total variation distance
    """

    value = 0.0 #The starting value is 0
    target_keys = target_distribution.distribution_dict.keys() #Get all bitstrings of first distribution
    measured_keys = measured_distribution.distribution_dict.keys() #Get all  bitstrings of second distribution
    all_keys = set(target_keys).union(measured_keys) #Combine all bitstrings together

    for bitstring in all_keys:
        #Get the probability of each bitstring for each distribution (if missing, it is 0)
        target_bitstring_value = target_distribution.distribution_dict.get(bitstring,0)
        measured_bitstring_value = measured_distribution.distribution_dict.get(bitstring,0)

        #The absolute value of the difference of all probabilities is the total variation distance
        value += abs(target_bitstring_value - measured_bitstring_value)

    return value
