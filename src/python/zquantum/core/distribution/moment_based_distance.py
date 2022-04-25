import math
from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    from zquantum.core.distribution import MeasurementOutcomeDistribution


def compute_moment_based_distance(
    target_distribution: "MeasurementOutcomeDistribution",
    measured_distribution: "MeasurementOutcomeDistribution",
    M: Optional[int] = 15
) -> float:
    """Compute the moment based distance between two distributions,
    potentially a target distribution and a measured distribution.

    Args:
        target_distribution: The target probability distribution.
        measured_distribution: The measured probability distribution.
        M: The number of iterations to reach convergence

    Returns:
        The value of the moment based distance
    """

    #The starting distance is 0
    distance = 0.0 
    #Get all bitstrings of first distribution
    target_keys = target_distribution.distribution_dict.keys() 
    #Get all  bitstrings of second distribution
    measured_keys = measured_distribution.distribution_dict.keys() 
    #Combine all bitstrings together
    all_keys = set(target_keys).union(measured_keys) 

    #Gamma parameter
    gamma = 2**len(measured_keys[0])

    #Go sufficiently many iterations to get accuracy
    for m in range(M):
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

        distance += 1/math.factorial(m) * s

    return distance
