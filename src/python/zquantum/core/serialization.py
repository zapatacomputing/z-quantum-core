"""Serialization module."""
import json
from operator import attrgetter
from typing import Any, Iterator
import numpy as np
from .bitstring_distribution import BitstringDistribution
from .utils import convert_array_to_dict, ValueEstimate, SCHEMA_VERSION


def preprocess(tree):
    """This inflates namedtuples into dictionaries, otherwise they would be serialized as lists.

    KJ: I found initial version of this code a while ago in a related SO question:
    https://stackoverflow.com/questions/43913256/understanding-subclassing-of-jsonencoder
    """
    if isinstance(tree, dict):
        return {k: preprocess(v) for k, v in tree.items()}
    elif isinstance(tree, tuple) and hasattr(tree, "_asdict"):
        return preprocess(tree._asdict())
    elif isinstance(tree, ValueEstimate):
        return tree.to_dict()
    elif isinstance(tree, (list, tuple)):
        return list(map(preprocess, tree))
    return tree


class ZapataEncoder(json.JSONEncoder):
    ENCODERS_TABLE = {
        np.ndarray: convert_array_to_dict,
        ValueEstimate: ValueEstimate.to_dict,
        BitstringDistribution: attrgetter("distribution_dict"),
    }

    def default(self, o: Any):
        if type(o) in self.ENCODERS_TABLE:
            return self.ENCODERS_TABLE[type(o)](o)
        return o

    def encode(self, o: Any):
        return super().encode(preprocess(o))

    def iterencode(self, o: Any, _one_shot: bool = ...) -> Iterator[str]:
        return super().iterencode(preprocess(o))


def save_optimization_results(optimization_results, filename):
    optimization_results["schema"] = SCHEMA_VERSION + "-optimization_result"
    with open(filename, "wt") as target_file:
        json.dump(optimization_results, target_file, cls=ZapataEncoder)
