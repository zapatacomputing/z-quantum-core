################################################################################
# © Copyright 2021 Zapata Computing Inc.
################################################################################
"""Definitions of keys for various symbol orderings."""
import re


def _convert_string_to_int_if_possible(text):
    return int(text) if text.isdigit() else text


def natural_key(symbol):
    """Convert symbol to a natural-ordering key.

    The natural ordering of symbols works as follows:
    1. Split a symbol on each group of digits.
    2. For each group in this split, convert all-digit groups to integers.

    Comparing such groups using lexicographical ordering gives natural ordering
    of symbols.

    In contrast to usual ordering obtained by casting symbol to string, natural
    ordering treats variable indices as integers. Thus, in this ordering,
    symbols beta_10, theta_2, beta_2, theta_1 would be ordered as follows:
    beta_2 < beta_10 < theta_1 < theta_2.

    The original idea behind this implementation was found byMSRudolph here:
    http://nedbatchelder.com/blog/200712/human_sorting.html
    """
    return [
        _convert_string_to_int_if_possible(group)
        for group in re.split(r"(\d+)", symbol.name)
    ]


def natural_key_revlex(symbol):
    """Convert symbol to lexically reversed natural-ordering key.

    This returns reversed key produced by `natural_key`. The main usage is for list
    of symbols with names of the form <symbol_name>_<number> where <symbol_name>
    can take one of several predefined values, and orders should consider <number>
    before <symbol_name>.

    For instance, given symbols beta_1, beta_2, theta_1, theta_2, sorting them
    using natural_key_revlex will give beta_1 < theta_1 < beta_2 < theta_2.
    """
    return list(reversed(natural_key(symbol)))


def natural_key_fixed_names_order(names_order):
    """Convert symbol to natural key but with custom ordering of names.

    Consider a QAOA ansatz in which parameters are naturally ordered as:
    gamma_0 < beta_0 < gamma_1 < beta_1 < ...

    The above is an example of natural_key_fixed_names_order in which name 'gamma'
    precedes name 'beta'.

    Note that unlike natural_key and natural_key_revlex, this function returns
    a key, i.e. it is a key factory.
    """
    symbol_weights = {name: i for i, name in enumerate(names_order)}

    def _key(symbol):
        name, index = symbol.name.split("_")
        return int(index), symbol_weights[name]

    return _key
