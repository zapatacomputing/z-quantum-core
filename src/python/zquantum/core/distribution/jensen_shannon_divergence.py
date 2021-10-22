from typing import TYPE_CHECKING, Dict

from .clipped_negative_log_likelihood import compute_clipped_negative_log_likelihood

if TYPE_CHECKING:
    from zquantum.core.distribution import DitSequenceDistribution


def compute_jensen_shannon_divergence(
    target_distribution: "DitSequenceDistribution",
    measured_distribution: "DitSequenceDistribution",
    distance_measure_parameters: Dict,
) -> float:
    """Computes the symmetrized version of the clipped negative log likelihood between a
     target distribution and a measured distribution.
    See Equation (4) in https://advances.sciencemag.org/content/5/10/eaaw9918?rss=1

    Args:
        target_distribution: The target probability distribution.
        measured_distribution: The measured probability distribution.

        distance_measure_parameters:
            epsilon (float): The small parameter needed to regularize log computation
            when argument is zero. The default value is 1e-9.

    Returns:
        float: The value of the symmetrized version
    """

    value = (
        compute_clipped_negative_log_likelihood(
            target_distribution, measured_distribution, distance_measure_parameters
        )
        / 2
        + compute_clipped_negative_log_likelihood(
            measured_distribution, target_distribution, distance_measure_parameters
        )
        / 2
    )

    return value
