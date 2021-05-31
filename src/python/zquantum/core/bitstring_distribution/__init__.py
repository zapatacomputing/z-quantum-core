import math  # This is needed for tests to work because of monkey patching

from ._bitstring_distribution import (
    BitstringDistribution,
    are_keys_binary_strings,
    create_bitstring_distribution_from_probability_distribution,
    evaluate_distribution_distance,
    is_bitstring_distribution,
    is_key_length_fixed,
    is_non_negative,
    is_normalized,
    load_bitstring_distribution,
    load_bitstring_distribution_set,
    normalize_bitstring_distribution,
    save_bitstring_distribution,
    save_bitstring_distribution_set,
)
from .distance_measures import (
    compute_clipped_negative_log_likelihood,
    compute_mmd,
    compute_multi_rbf_kernel,
    compute_rbf_kernel,
)
