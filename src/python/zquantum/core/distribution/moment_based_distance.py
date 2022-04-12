import math
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from zquantum.core.distribution import MeasurementOutcomeDistribution


def compute_moment_based_distance(
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

    #Gamma parameter (may need to be reversed)
    gamma = int(max(all_keys),2) #max value measured

    #Go sufficiently many iterations to get accuracy
    for m in range(15):
        s = 0
        for bitstring in all_keys:
            x = int(bitstring,2)
            scaler = (x/gamma)**m

            target_bitstring_value = target_distribution.distribution_dict.get(
                bitstring,0
            )
            measured_bitstring_value = measured_distribution.distribution_dict.get(
                bitstring,0
            )

            diff = target_bitstring_value - measured_bitstring_value

            s += abs(scaler*diff)

        value += 1/math.factorial(m) * s

    return value
