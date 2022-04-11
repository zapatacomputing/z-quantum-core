import math
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from zquantum.core.distribution import MeasurementOutcomeDistribution


def compute_total_variation_distance(
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

    #The starting value is 0
    value = 0.0 
    #Get all bitstrings of first distribution
    target_keys = target_distribution.distribution_dict.keys() 
    #Get all  bitstrings of second distribution
    measured_keys = measured_distribution.distribution_dict.keys() 
    #Combine all bitstrings together
    all_keys = set(target_keys).union(measured_keys) 

    for bitstring in all_keys:
        #Get the probability of each bitstring for each distribution (if missing, it is 0)
        target_bitstring_value = target_distribution.distribution_dict.get(
            bitstring,0
        )
        measured_bitstring_value = measured_distribution.distribution_dict.get(
            bitstring,0
        )

        #Sum of abs of all probability differences
        value += abs(target_bitstring_value - measured_bitstring_value)

    return value
