"""
This file is needed for backward-compatibility, due to the changing of
the bitstring_distribution module to simply distribution module. This change
was driven by our need to generalize the quantum systems we want to support,
by allowing for abitrary-level systems.
"""
# Distance Measures
from .distribution import (
    DitSequenceDistribution,
    are_keys_non_negative_integer_tuples,
    are_non_tuple_keys_valid_binary_strings,
    change_tuple_dict_keys_to_comma_separated_digitstrings,
    compute_clipped_negative_log_likelihood,
    compute_jensen_shannon_divergence,
    compute_mmd,
    compute_multi_rbf_kernel,
    compute_rbf_kernel,
    create_bitstring_distribution_from_probability_distribution,
    evaluate_distribution_distance,
    is_ditsequence_distribution,
    is_key_length_fixed,
    is_non_negative,
    is_normalized,
    load_ditsequence_distribution,
    load_ditsequence_distributions,
    normalize_ditstring_distribution,
    preprocess_distibution_dict,
    save_ditsequence_distribution,
    save_ditsequence_distributions,
)
