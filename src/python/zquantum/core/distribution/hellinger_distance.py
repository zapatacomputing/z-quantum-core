import math
from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    from zquantum.core.distribution import MeasurementOutcomeDistribution


def compute_hellinger_distance(
    target_distribution: "MeasurementOutcomeDistribution",
    measured_distribution: "MeasurementOutcomeDistribution"
) -> float:
    """Compute the hellinger distance between two distributions,
    potentially a target distribution and a measured distribution.

    Args:
        target_distribution: The target probability distribution.
        measured_distribution: The measured probability distribution.

    Returns:
        The value of the hellinger distance
    """
    #Get all bitstrings of first distribution
    target_keys = target_distribution.distribution_dict.keys() 
    #Get all  bitstrings of second distribution
    measured_keys = measured_distribution.distribution_dict.keys() 
    #Combine all bitstrings together
    all_keys = set(target_keys).union(measured_keys) 

    s = 0
    for bitstring in all_keys:
        p = target_distribution.distribution_dict.get(
                bitstring,0
            )
        q = measured_distribution.distribution_dict.get(
                bitstring,0
            )
        s += (math.sqrt(p)-math.sqrt(q))**2

    return(1/math.sqrt(2)*math.sqrt(s))